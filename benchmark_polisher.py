"""Benchmark the voice polisher against GMI Cloud models.

Modes:
  (default)  Full run: 12 cases x 3 models, 3 determinism repeats on Gemini, and the
             offline fail-closed suite. Writes outputs/run_<timestamp>.jsonl (every
             row, including raw output) and benchmark_results.json (summary).
  --probe    One short request per model, to see finish_reason and reasoning fields.
             Writes outputs/probe_<timestamp>.json and no summary.
  --dry-run  No network. Prints the request plan and cost ceilings per model.

The API key is read from GMI_API_KEY. It is never printed or written to disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import voice_polisher as vp

HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "test_cases.json"
PRICING_PATH = HERE / "pricing_snapshot.json"
RESULTS_PATH = HERE / "benchmark_results.json"
OUTPUT_DIR = HERE / "outputs"

MODELS = ["google/gemini-3.8-flash", "zai-org/GLM-5.3-Flash", "deepseek-ai/DeepSeek-V4-Flash"]
DETERMINISM_MODEL = "google/gemini-3.8-flash"
DETERMINISM_CASES = ["c03", "c05", "c11"]
DETERMINISM_RUNS = 3
EXAMPLE_CASES = ["c01", "c03", "c06", "c11"]
PROBE_TEXT = "this is a probe of the voice polisher so please just repeat it back cleaned up"
# Worst case per chunk: the first request truncates, both halves truncate, and all four
# quarters succeed. That is 1 + 2 + 4 = 7 requests, which is the most the code can send.
RETRY_REQUESTS_PER_CHUNK = 2 ** (vp.MAX_SPLIT_DEPTH + 1) - 1
# Conservative token estimate for ceilings only. English is closer to 4 characters per token.
CHARS_PER_TOKEN_CEILING = 3
OFFLINE_KEY = "offline-test-key"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_cases() -> list[dict[str, Any]]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if len(cases) != 12:
        raise SystemExit(f"expected 12 cases in {CASES_PATH.name}, found {len(cases)}")
    return cases


def load_pricing() -> dict[str, Any]:
    return json.loads(PRICING_PATH.read_text(encoding="utf-8"))


def cost_of(usage: vp.Usage, price: dict[str, Any]) -> tuple[float, float]:
    """(list USD, after-discount USD). Discount rule matches day2: effective = list * (1 - discount_to_user)."""
    list_usd = (
        usage.prompt_tokens / 1e6 * price["prompt_per_1m_usd"]
        + usage.completion_tokens / 1e6 * price["completion_per_1m_usd"]
    )
    return list_usd, list_usd * (1 - price["discount_to_user"])


def check_output(output: str, checks: dict[str, Any]) -> list[str]:
    if not output.strip():
        return ["empty output"]
    lowered = output.lower()
    failures = []
    for item in checks.get("contains", []):
        if item not in output:
            failures.append(f"missing {item!r}")
    for item in checks.get("contains_ci", []):
        if item.lower() not in lowered:
            failures.append(f"missing (ignore case) {item!r}")
    for item in checks.get("not_contains_ci", []):
        if item.lower() in lowered:
            failures.append(f"contains {item!r}")
    for item in checks.get("not_equal_ci", []):
        if output.strip().lower() == item.lower():
            failures.append(f"equals {item!r}")
    ends = checks.get("ends_with")
    if ends and not output.rstrip().endswith(ends):
        failures.append(f"does not end with {ends!r}")
    return failures


def latency_stats(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0, "median": None, "p95": None, "mean": None}
    ordered = sorted(samples)
    nearest_rank_p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 3),
        "p95": round(nearest_rank_p95, 3),
        "mean": round(statistics.fmean(ordered), 3),
    }


def _parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _reply_shape(data: Any) -> dict[str, Any] | None:
    """Field shape of a reply (no content text), for probing."""
    if not isinstance(data, dict):
        return None
    try:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content")
        return {
            "finish_reason": choice.get("finish_reason"),
            "message_keys": sorted(message.keys()),
            "content_chars": len(message.get("content") or ""),
            "reasoning_chars": None if reasoning is None else len(reasoning),
            "usage": data.get("usage"),
        }
    except (KeyError, IndexError, TypeError, AttributeError):
        return {"unexpected_shape": True}


class TimedTransport:
    """Wraps the real HTTP transport. Records status and wall time per request, and sums usage from every reply."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.usage = vp.Usage()
        self.last_reply: dict[str, Any] | None = None

    def __call__(self, url, headers, body, timeout_s):
        started = time.perf_counter()
        try:
            status, raw = vp._urllib_transport(url, headers, body, timeout_s)
        except BaseException as exc:
            self.samples.append({"status": None, "latency_s": round(time.perf_counter() - started, 3), "error": type(exc).__name__})
            raise
        self.samples.append({"status": status, "latency_s": round(time.perf_counter() - started, 3)})
        data = _parse_json(raw)
        if isinstance(data, dict):
            self.usage.add(data.get("usage"))
        self.last_reply = _reply_shape(data)
        return status, raw


def run_one(polisher: vp.VoicePolisher, transport: TimedTransport, case: dict[str, Any], *,
            phase: str, model: str, price: dict[str, Any], run_id: str, run_index: int) -> dict[str, Any]:
    transport.reset()
    error, output = None, ""
    try:
        output = polisher.polish(case["transcript"]).text
    except vp.PolisherError as exc:
        error = str(exc)
    failures = [f"error: {error}"] if error else check_output(output, case["checks"])
    list_usd, effective_usd = cost_of(transport.usage, price)
    return {
        "run_id": run_id, "phase": phase, "model": model, "case_id": case["id"],
        "category": case["category"], "run_index": run_index,
        "chunks": len(vp.chunk_words(case["transcript"], vp.DEFAULT_CHUNK_WORDS)),
        "ok": not failures, "failures": failures, "error": error, "output": output,
        "http_requests": len(transport.samples),
        "statuses": [s["status"] for s in transport.samples],
        "latencies_s": [s["latency_s"] for s in transport.samples],
        "prompt_tokens": transport.usage.prompt_tokens,
        "completion_tokens": transport.usage.completion_tokens,
        "list_usd": list_usd, "effective_usd": effective_usd,
    }


def ceiling(model: str, transcripts: list[str], retry: bool, dictionary: vp.Dictionary,
            price: dict[str, Any]) -> dict[str, Any]:
    """Upper bound on requests and spend. Every reply is assumed to hit max_tokens."""
    per_chunk = RETRY_REQUESTS_PER_CHUNK if retry else 1
    requests = prompt = completion = 0
    for transcript in transcripts:
        for chunk in vp.chunk_words(transcript, vp.DEFAULT_CHUNK_WORDS):
            body = vp.build_request_body(model, vp.build_user_message(chunk, dictionary.hints_for(chunk)), vp.DEFAULT_MAX_TOKENS)
            prompt_one = math.ceil(len(json.dumps(body["messages"], ensure_ascii=False)) / CHARS_PER_TOKEN_CEILING)
            requests += per_chunk
            prompt += per_chunk * prompt_one
            completion += per_chunk * vp.DEFAULT_MAX_TOKENS
    list_usd = prompt / 1e6 * price["prompt_per_1m_usd"] + completion / 1e6 * price["completion_per_1m_usd"]
    return {
        "requests": requests, "prompt_tokens": prompt, "completion_tokens": completion,
        "list_usd": list_usd, "after_discount_usd": list_usd * (1 - price["discount_to_user"]),
    }


def add_ceilings(parts: list[dict[str, Any]], times: int = 1) -> dict[str, Any]:
    total = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "list_usd": 0.0, "after_discount_usd": 0.0}
    for part in parts:
        for key in total:
            total[key] += part[key] * times
    return total


def dry_run(models: list[str]) -> int:
    cases = load_cases()
    pricing = load_pricing()["models"]
    dictionary = vp.Dictionary.load(vp.DICTIONARY_PATH)
    transcripts = [c["transcript"] for c in cases]
    det = [c["transcript"] for c in cases if c["id"] in DETERMINISM_CASES]
    probe = [PROBE_TEXT]
    print(f"Dry run. No requests sent. Ceilings assume every reply hits max_tokens={vp.DEFAULT_MAX_TOKENS}.")
    print(f"Worst case allows {RETRY_REQUESTS_PER_CHUNK} requests per chunk (split depth {vp.MAX_SPLIT_DEPTH}).\n")
    header = f"{'model':34} {'req(1st)':>8} {'req(worst)':>10} {'list $ 1st':>11} {'list $ worst':>13} {'after $ worst':>14}"
    print(header)
    print("-" * len(header))
    grand_first, grand_worst = [], []
    for model in models:
        price = pricing[model]
        first = add_ceilings([ceiling(model, transcripts, False, dictionary, price)])
        worst = add_ceilings([ceiling(model, transcripts, True, dictionary, price)])
        grand_first.append(first)
        grand_worst.append(worst)
        print(f"{model:34} {first['requests']:>8} {worst['requests']:>10} {first['list_usd']:>11.5f} {worst['list_usd']:>13.5f} {worst['after_discount_usd']:>14.5f}")
    det_worst = add_ceilings([ceiling(DETERMINISM_MODEL, det, True, dictionary, pricing[DETERMINISM_MODEL])], DETERMINISM_RUNS)
    probe_worst = add_ceilings([ceiling(m, probe, True, dictionary, pricing[m]) for m in models])
    sweep_first = add_ceilings(grand_first)
    sweep_worst = add_ceilings(grand_worst)
    total_worst_list = sweep_worst["list_usd"] + det_worst["list_usd"] + probe_worst["list_usd"]
    total_worst_after = sweep_worst["after_discount_usd"] + det_worst["after_discount_usd"] + probe_worst["after_discount_usd"]
    print("-" * len(header))
    print(f"{'sweep total':34} {sweep_first['requests']:>8} {sweep_worst['requests']:>10} {sweep_first['list_usd']:>11.5f} {sweep_worst['list_usd']:>13.5f} {sweep_worst['after_discount_usd']:>14.5f}")
    print(f"\nDeterminism ({DETERMINISM_MODEL}, {DETERMINISM_CASES}, {DETERMINISM_RUNS} runs): worst-case list ${det_worst['list_usd']:.5f}")
    print(f"Probe (one short request per model): worst-case list ${probe_worst['list_usd']:.5f}")
    print(f"TOTAL WORST-CASE CEILING: list ${total_worst_list:.5f}, after discount ${total_worst_after:.5f}")
    print(f"First-attempt total (no retries, every reply at max_tokens): list ${sweep_first['list_usd']:.5f}")
    print("\nDry run wrote no files.")
    return 0


PROBE_LABEL = "probe"


def run_probe(models: list[str], api_key: str) -> int:
    dictionary = vp.Dictionary.load(vp.DICTIONARY_PATH)
    pricing = load_pricing()["models"]
    run_id = utc_stamp()
    records = []
    for model in models:
        transport = TimedTransport()
        polisher = vp.VoicePolisher(api_key=api_key, model=model, dictionary=dictionary, transport=transport)
        error, output = None, ""
        try:
            output = polisher.polish(PROBE_TEXT).text
        except vp.PolisherError as exc:
            error = str(exc)
        list_usd, effective_usd = cost_of(transport.usage, pricing[model])
        record = {
            "model": model, "ok": error is None, "error": error, "output": output,
            "statuses": [s["status"] for s in transport.samples],
            "latencies_s": [s["latency_s"] for s in transport.samples],
            "reply_shape": transport.last_reply,
            "prompt_tokens": transport.usage.prompt_tokens,
            "completion_tokens": transport.usage.completion_tokens,
            "list_usd": list_usd, "effective_usd": effective_usd,
        }
        records.append(record)
        shape = record["reply_shape"] or {}
        print(f"{model}")
        print(f"  ok={record['ok']} error={error!r} statuses={record['statuses']} latency_s={record['latencies_s']}")
        print(f"  finish_reason={shape.get('finish_reason')} message_keys={shape.get('message_keys')}")
        print(f"  content_chars={shape.get('content_chars')} reasoning_chars={shape.get('reasoning_chars')} usage={shape.get('usage')}")
        print(f"  output={output!r}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"probe_{run_id}.json"
    path.write_text(json.dumps({"run_id": run_id, "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


FAIL_CLOSED_MESSAGE = '{"error": {"message": "upstream unavailable"}}'


def _reply_bytes(content: str = "", finish: str = "stop", reasoning: str | None = None) -> bytes:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    body = {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return json.dumps(body).encode("utf-8")


def _raise_timeout(*_args: Any) -> None:
    raise TimeoutError("simulated timeout")


FAIL_CLOSED_MODES: dict[str, Callable[..., Any]] = {
    "http_500": lambda *_: (500, FAIL_CLOSED_MESSAGE.encode("utf-8")),
    "timeout": _raise_timeout,
    "empty_content": lambda *_: (200, _reply_bytes("")),
    "reasoning_only": lambda *_: (200, _reply_bytes("", reasoning="Let me think about the cleanup first")),
    "unrecoverable_truncation": lambda *_: (200, _reply_bytes("", finish="length")),
}


class ScriptedFailure:
    def __init__(self, responder: Callable[..., Any]) -> None:
        self.responder = responder
        self.calls = 0

    def __call__(self, url, headers, body, timeout_s):
        self.calls += 1
        return self.responder(url, headers, body, timeout_s)


def run_fail_closed(dictionary: vp.Dictionary, transcript: str) -> list[dict[str, Any]]:
    results = []
    for mode, responder in FAIL_CLOSED_MODES.items():
        transport = ScriptedFailure(responder)
        polisher = vp.VoicePolisher(api_key=OFFLINE_KEY, model=DEFAULT_MODEL_NAME, dictionary=dictionary, transport=transport)
        try:
            text = polisher.polish(transcript).text
            outcome, message = "returned_text", text
        except vp.PolisherError as exc:
            outcome, message = "error", str(exc)
        results.append({"mode": mode, "passed": outcome == "error", "outcome": outcome,
                        "error": message, "requests": transport.calls})
    return results


DEFAULT_MODEL_NAME = vp.DEFAULT_MODEL


def summarize_model(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sweep = [r for r in rows if r["phase"] == "sweep" and r["model"] == model]
    prompt = sum(r["prompt_tokens"] for r in sweep)
    completion = sum(r["completion_tokens"] for r in sweep)
    list_usd = sum(r["list_usd"] for r in sweep)
    effective_usd = sum(r["effective_usd"] for r in sweep)
    return {
        "cases_passed": sum(r["ok"] for r in sweep),
        "cases_total": len(sweep),
        "failed_cases": [r["case_id"] for r in sweep if not r["ok"]],
        "http_requests": sum(r["http_requests"] for r in sweep),
        "latency_per_request_s": latency_stats([x for r in sweep for x in r["latencies_s"]]),
        "tokens": {"prompt": prompt, "completion": completion, "total": prompt + completion},
        "cost_usd": {"list": round(list_usd, 6), "after_discount": round(effective_usd, 6)},
    }


def summarize_determinism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    det = [r for r in rows if r["phase"] == "determinism"]
    by_case: dict[str, list[str]] = {}
    checks_ok: dict[str, int] = {}
    for r in det:
        by_case.setdefault(r["case_id"], []).append(r["output"])
        checks_ok[r["case_id"]] = checks_ok.get(r["case_id"], 0) + int(r["ok"])
    identical = {cid: len(set(outs)) == 1 for cid, outs in by_case.items()}
    return {
        "model": DETERMINISM_MODEL,
        "runs_per_case": DETERMINISM_RUNS,
        "identical_across_runs": identical,
        "checks_passed_per_case": checks_ok,
        "score": f"{sum(identical.values())} of {len(identical)} cases identical",
    }


def print_table(results: dict[str, Any]) -> None:
    print("\nPer model (12 sweep cases):")
    for model, s in results["models"].items():
        lat = s["latency_per_request_s"]
        print(f"  {model}")
        print(f"    passed {s['cases_passed']}/{s['cases_total']}  failed: {s['failed_cases'] or 'none'}")
        print(f"    latency per request s: median {lat['median']}  p95 {lat['p95']}  mean {lat['mean']}  (n={lat['n']})")
        print(f"    tokens: {s['tokens']}  http requests: {s['http_requests']}")
        print(f"    cost USD: list {s['cost_usd']['list']}  after discount {s['cost_usd']['after_discount']}")
    det = results["determinism"]
    print(f"\nDeterminism ({det['model']}, {det['runs_per_case']} runs each): {det['score']}")
    print("Fail-closed suite (offline):")
    for row in results["fail_closed"]["modes"]:
        print(f"  {row['mode']:26} {'PASS' if row['passed'] else 'FAIL'}  {row['outcome']}  {row['error']!r}")
    print(f"  score: {results['fail_closed']['score']}")
    print("\nExamples (verbatim, Gemini):")
    for ex in results["examples"]:
        print(f"  [{ex['case_id']} {ex['category']}] requests={ex['requests']} latency_s={ex['latencies_s']}")
        print(f"    before: {ex['before']}")
        print(f"    after:  {ex['after']}")


def full_run(models: list[str], api_key: str) -> int:
    cases = load_cases()
    pricing = load_pricing()["models"]
    dictionary = vp.Dictionary.load(vp.DICTIONARY_PATH)
    run_id = utc_stamp()
    rows: list[dict[str, Any]] = []
    by_id = {c["id"]: c for c in cases}

    for model in models:
        transport = TimedTransport()
        polisher = vp.VoicePolisher(api_key=api_key, model=model, dictionary=dictionary, transport=transport)
        for case in cases:
            rows.append(run_one(polisher, transport, case, phase="sweep", model=model,
                                price=pricing[model], run_id=run_id, run_index=1))

    transport = TimedTransport()
    polisher = vp.VoicePolisher(api_key=api_key, model=DETERMINISM_MODEL, dictionary=dictionary, transport=transport)
    for case_id in DETERMINISM_CASES:
        for run_index in range(1, DETERMINISM_RUNS + 1):
            rows.append(run_one(polisher, transport, by_id[case_id], phase="determinism", model=DETERMINISM_MODEL,
                                price=pricing[DETERMINISM_MODEL], run_id=run_id, run_index=run_index))

    fail_closed = run_fail_closed(dictionary, by_id["c05"]["transcript"])

    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_path = OUTPUT_DIR / f"run_{run_id}.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    gemini_rows = {r["case_id"]: r for r in rows if r["phase"] == "sweep" and r["model"] == DETERMINISM_MODEL}
    examples = [{
        "case_id": cid, "category": by_id[cid]["category"], "before": by_id[cid]["transcript"],
        "after": gemini_rows[cid]["output"], "ok": gemini_rows[cid]["ok"],
        "requests": gemini_rows[cid]["http_requests"], "latencies_s": gemini_rows[cid]["latencies_s"],
    } for cid in EXAMPLE_CASES]

    transcripts = [c["transcript"] for c in cases]
    det_transcripts = [by_id[cid]["transcript"] for cid in DETERMINISM_CASES]
    worst = add_ceilings(
        [ceiling(m, transcripts, True, dictionary, pricing[m]) for m in models]
    )
    det_worst = add_ceilings([ceiling(DETERMINISM_MODEL, det_transcripts, True, dictionary, pricing[DETERMINISM_MODEL])], DETERMINISM_RUNS)
    probe_worst = add_ceilings([ceiling(m, [PROBE_TEXT], True, dictionary, pricing[m]) for m in models])
    ceiling_total = {
        "list_usd": round(worst["list_usd"] + det_worst["list_usd"] + probe_worst["list_usd"], 6),
        "after_discount_usd": round(worst["after_discount_usd"] + det_worst["after_discount_usd"] + probe_worst["after_discount_usd"], 6),
    }

    results = {
        "schema_version": 1,
        "run_id": run_id,
        "scope": "GMI-hosted models only, not the whole market.",
        "pricing": {
            "captured_at": load_pricing()["captured_at"],
            "note": "Per-1M prices from pricing_snapshot.json (verified against live GMI MCP catalog reads).",
        },
        "config": {
            "temperature": 0.0, "max_tokens": vp.DEFAULT_MAX_TOKENS, "chunk_words": vp.DEFAULT_CHUNK_WORDS,
            "timeout_s": vp.DEFAULT_TIMEOUT_S, "max_split_depth": vp.MAX_SPLIT_DEPTH,
            "reasoning_effort": "low (google/gemini models only)", "stream": False,
        },
        "ceiling_worst_case_usd": ceiling_total,
        "models": {m: summarize_model(m, rows) for m in models},
        "determinism": summarize_determinism(rows),
        "fail_closed": {
            "score": f"{sum(r['passed'] for r in fail_closed)} of {len(fail_closed)}",
            "modes": fail_closed,
        },
        "cases": [
            {"id": r["case_id"], "category": r["category"], "model": r["model"], "ok": r["ok"], "failures": r["failures"]}
            for r in rows if r["phase"] == "sweep"
        ],
        "examples": examples,
        "raw_log": str(raw_path.relative_to(HERE)),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"run_id {run_id}")
    print(f"raw rows: {raw_path}")
    print(f"summary:  {RESULTS_PATH}")
    print_table(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the voice polisher against GMI Cloud models.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="no network; print ceilings only")
    mode.add_argument("--probe", action="store_true", help="one short request per model")
    parser.add_argument("--models", nargs="+", default=MODELS, help="models to run")
    args = parser.parse_args(argv)
    if args.dry_run:
        return dry_run(args.models)
    api_key = os.environ.get(vp.API_KEY_ENV, "")
    if not api_key:
        print(f"error: {vp.API_KEY_ENV} is not set", file=sys.stderr)
        return 2
    if args.probe:
        return run_probe(args.models, api_key)
    return full_run(args.models, api_key)


if __name__ == "__main__":
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
