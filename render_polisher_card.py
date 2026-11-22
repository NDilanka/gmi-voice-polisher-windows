#!/usr/bin/env python3
"""Render the minimal Post 2 editorial card: a 1920x1080 (16:9) PNG.

Every figure and both verbatim before/after pairs are read at run time from:
  benchmark_results.json   model table, fail-closed score, DeepSeek cap count, config
  test_cases.json          raw ASR text per case and the case count
  outputs/run_<id>.jsonl   polished Gemini output per case (path from benchmark_results.json "raw_log")

Nothing numeric is typed in by hand. The script stops with an error instead of guessing.
Usage: py render_polisher_card.py
"""
import html
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "benchmark_results.json"
TEST_CASES_JSON = ROOT / "test_cases.json"
OUT_PNG = ROOT / "polisher-card.png"
FONT_FILE = Path(
    os.environ.get("XPOST_INTER_FONT")
    or Path.home() / ".factory" / "skills" / "x-post" / "templates" / "fonts" / "inter-var.woff2"
)

GEMINI_ID = "google/gemini-3.8-flash"
DEEPSEEK_ID = "deepseek-ai/DeepSeek-V4-Flash"
LIVE_CASE = "c02"  # raw: "we run phonon with parakeet on the mac and sherpa-onnx on linux"
SELF_CASE = "c06"  # raw: "send the summary to Alice wait no send it to Bob"

# Pipeline labels are design copy, not benchmark data.
PIPELINE = [
    ("01", "Mic", "Windows WASAPI", False),
    ("02", "Local ASR", "Parakeet TDT 0.6B v2", False),
    ("03", "Dictionary", "dictionary.json hints", False),
    ("04", "GMI Cloud", "Temp 0 · Fail-closed", True),
]

CSS = """
:root {
  --paper: #F5F3EE;
  --ink: #141414;
  --stone-700: #57534E;
  --stone-500: #8A857C;
  --rule: #DCD7CE;
  --tint: #EFECE6;
  --accent: #D94E1F;
  --sans: "CardInter", Inter, "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", "Cascadia Code", Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  width: 1920px; height: 1080px; overflow: hidden;
  background: var(--paper); color: var(--ink);
  font-family: var(--sans); -webkit-font-smoothing: antialiased;
}
.page { width: 1920px; height: 1080px; padding: 96px; display: flex; flex-direction: column; }
.mono { font-family: var(--mono); }

/* Header */
.eyebrow { font-size: 18px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); line-height: 1.2; }
h1 { margin-top: 16px; font-size: 72px; font-weight: 800; letter-spacing: -0.03em; line-height: 1.04; white-space: nowrap; }

/* Pipeline: hairline connector, terracotta only on the cloud step */
.pipeline { margin-top: 36px; display: grid; grid-template-columns: repeat(4, 1fr); }
.step { position: relative; padding: 20px 32px 0 0; }
.step::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 1px; background: var(--rule); }
.step.hot::before { top: -1px; height: 3px; background: var(--accent); }
.step-no { font-family: var(--mono); font-size: 16px; letter-spacing: 0.06em; color: var(--stone-500); }
.step.hot .step-no { color: var(--accent); font-weight: 700; }
.step-name { margin-top: 10px; font-size: 32px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; }
.step-detail { margin-top: 8px; font-family: var(--mono); font-size: 18px; color: var(--stone-700); }

/* Main 2-column split: about 60% benchmark, about 36% verbatim */
.main { margin-top: 40px; display: grid; grid-template-columns: 1040px 624px; column-gap: 64px; }

/* Benchmark table (CSS grid, no heavy boxes) */
.bench-head, .row { display: grid; grid-template-columns: 1fr 170px 170px 210px; align-items: center; }
.bench-head {
  padding: 0 24px 14px 28px; border-bottom: 1px solid var(--ink);
  font-size: 15px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--stone-500);
}
.row { position: relative; height: 128px; padding: 0 24px 0 28px; border-bottom: 1px solid var(--rule); }
.row.hot { background: var(--tint); }
.row.hot::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--accent); }
.bench .num { text-align: right; }
.model { font-family: var(--mono); font-size: 24px; font-weight: 500; }
.row .num { font-family: var(--mono); font-size: 34px; font-weight: 500; }
.row.hot .model, .row.hot .num { font-weight: 700; }
.bench-note { margin-top: 26px; font-size: 20px; line-height: 1.4; color: var(--stone-700); }
.bench-note .mono { color: var(--ink); }

/* Verbatim panel: hairline architectural frame */
.verbatim { border: 1px solid var(--stone-500); padding: 28px 40px; display: flex; flex-direction: column; }
.eyebrow-sm { font-size: 16px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: var(--stone-700); line-height: 1.2; }
.vb-label { margin-top: 20px; font-size: 16px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--stone-500); line-height: 1.2; }
.raw { margin-top: 10px; font-size: 24px; line-height: 1.3; color: var(--stone-500); }
.polished { margin-top: 10px; font-size: 28px; line-height: 1.25; font-weight: 700; letter-spacing: -0.01em; }
.turn { display: flex; align-items: center; gap: 16px; margin: 16px 0; }
.turn-arrow { flex: 0 0 20px; width: 20px; height: 28px; }
.hair { flex: 1; height: 1px; background: var(--rule); }
.mini { margin-top: 20px; padding-top: 18px; border-top: 1px solid var(--rule); }
.mini .vb-label { margin-top: 0; }
.mini-raw { margin-top: 6px; font-size: 20px; line-height: 1.3; color: var(--stone-500); }
.mini-arrow { display: block; width: 14px; height: 20px; margin: 6px 0; }
.mini-pol { font-size: 22px; line-height: 1.3; font-weight: 700; }

/* Footer */
.foot {
  margin-top: auto; padding-top: 18px; border-top: 1px solid var(--rule);
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 18px; color: var(--stone-500);
}
"""

# Runs in the page. Flags anything outside the 96 px margin or any text overflow.
QA_JS = """() => {
  const issues = [];
  const left = 96, top = 96, right = 1824, bottom = 984;
  document.querySelectorAll('.page *').forEach((el) => {
    if (el.closest('svg') && el.tagName.toLowerCase() !== 'svg') return;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const cls = (typeof el.className === 'string' && el.className) ? '.' + el.className.trim().split(/\\s+/).join('.') : '';
    const name = el.tagName.toLowerCase() + cls;
    if (r.left < left - 0.5 || r.top < top - 0.5 || r.right > right + 0.5 || r.bottom > bottom + 0.5) {
      issues.push('outside 96px margin: ' + name + ' [' + r.left.toFixed(0) + ',' + r.top.toFixed(0) + ' - ' + r.right.toFixed(0) + ',' + r.bottom.toFixed(0) + ']');
    }
    if (el.clientWidth > 0 && el.scrollWidth > el.clientWidth + 1) issues.push('horizontal overflow: ' + name);
    // Scroll overhang on visible-overflow boxes is line-height slack, not clipping; flag only clipped boxes.
    if (getComputedStyle(el).overflowY !== 'visible' && el.clientHeight > 0 && el.scrollHeight > el.clientHeight + 1) issues.push('vertical clip: ' + name);
  });
  return issues;
}"""


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def esc(value):
    return html.escape(str(value), quote=True)


def quote(value):
    return "\u201c" + esc(value) + "\u201d"


def arrow(cls):
    return (
        f'<svg class="{cls}" viewBox="0 0 20 28" aria-hidden="true">'
        '<path d="M10 1 V25 M3 18 L10 25 L17 18" fill="none" stroke="#D94E1F" '
        'stroke-width="2" stroke-linecap="square"/></svg>'
    )


def load_data():
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    cases = json.loads(TEST_CASES_JSON.read_text(encoding="utf-8"))["cases"]
    case_text = {c["id"]: c["transcript"] for c in cases}

    models = results["models"]
    for required in (GEMINI_ID, DEEPSEEK_ID):
        if required not in models:
            fail(f"{required} missing from benchmark_results.json")
    for model_id, m in models.items():
        if m["cases_total"] != len(cases):
            fail(f"{model_id} cases_total {m['cases_total']} != {len(cases)} test cases")

    bench = []
    for model_id, m in models.items():
        after = m["cost_usd"]["after_discount"]
        listed = m["cost_usd"]["list"]
        bench.append({
            "model": model_id,
            "passed": f"{m['cases_passed']} / {m['cases_total']}",
            "median": f"{m['latency_per_request_s']['median']:.2f} s",
            "cost": f"${after:.4f}" + ("*" if after < listed - 1e-12 else ""),
            "hot": model_id == GEMINI_ID,
        })

    fc = results["fail_closed"]
    fc_total = len(fc["modes"])
    fc_passed = sum(1 for mode in fc["modes"] if mode["passed"])
    if fc["score"] != f"{fc_passed} of {fc_total}":
        fail(f"fail_closed score '{fc['score']}' does not match modes ({fc_passed} of {fc_total})")

    cap_n = sum(
        1 for c in results["cases"]
        if c["model"] == DEEPSEEK_ID and not c["ok"] and any("max_tokens" in f for f in c["failures"])
    )

    cfg = results["config"]
    temp = f"{cfg['temperature']:g}"
    chunk_words = cfg["chunk_words"]
    max_tokens = cfg["max_tokens"]
    gem_name = GEMINI_ID.split("/")[-1].replace("-", " ").upper()

    raw_log = ROOT / results["raw_log"].replace("\\", "/")
    log_rows = [json.loads(line) for line in raw_log.read_text(encoding="utf-8").splitlines() if line.strip()]

    def polished(model_id, case_id):
        hits = [r for r in log_rows if r["phase"] == "sweep" and r["model"] == model_id and r["case_id"] == case_id]
        if len(hits) != 1:
            fail(f"expected 1 sweep row for {model_id} {case_id}, found {len(hits)}")
        if not hits[0]["ok"] or not hits[0]["output"]:
            fail(f"sweep row for {model_id} {case_id} is not a clean pass")
        return hits[0]["output"]

    return {
        "bench": bench,
        "fc_passed": fc_passed,
        "fc_total": fc_total,
        "cap_n": cap_n,
        "max_tokens": max_tokens,
        "n_cases": len(cases),
        "chunk_words": chunk_words,
        "temp": temp,
        "gem_name": gem_name,
        "live_raw": case_text[LIVE_CASE],
        "live_out": polished(GEMINI_ID, LIVE_CASE),
        "self_raw": case_text[SELF_CASE],
        "self_out": polished(GEMINI_ID, SELF_CASE),
        "foot_left": f"GMI-hosted models only, not the whole market · {len(cases)} dictation cases · *after account discount",
        "foot_right": f"Temp {temp} · {chunk_words}-word chunks · Zero silent fallback",
    }


def build_html(d, font_face):
    steps = []
    for num, name, detail, hot in PIPELINE:
        cls = "step hot" if hot else "step"
        steps.append(
            f'<div class="{cls}">'
            f'<div class="step-no">{num}</div>'
            f'<div class="step-name">{esc(name)}</div>'
            f'<div class="step-detail">{esc(detail)}</div>'
            "</div>"
        )
    steps_html = "".join(steps)

    rows = []
    for b in d["bench"]:
        cls = "row hot" if b["hot"] else "row"
        rows.append(
            f'<div class="{cls}">'
            f'<span class="model">{esc(b["model"])}</span>'
            f'<span class="num">{esc(b["passed"])}</span>'
            f'<span class="num">{esc(b["median"])}</span>'
            f'<span class="num">{esc(b["cost"])}</span>'
            "</div>"
        )
    rows_html = "\n      ".join(rows)

    note = (
        f'<span class="mono">{d["fc_passed"]} / {d["fc_total"]}</span> fail-closed safety checks passed'
        f' · DeepSeek hit the <span class="mono">{d["max_tokens"]}</span>-token cap'
        f' on <span class="mono">{d["cap_n"]}</span> cases'
    )

    n_cases = d["n_cases"]
    eyebrow_live = f"LIVE SAMPLE · {esc(d['gem_name'])} (TEMP {esc(d['temp'])})"
    live_raw = quote(d["live_raw"])
    live_out = quote(d["live_out"])
    self_raw = quote(d["self_raw"])
    self_out = quote(d["self_out"])
    arrow_big = arrow("turn-arrow")
    arrow_small = arrow("mini-arrow")
    foot_left = esc(d["foot_left"])
    foot_right = esc(d["foot_right"])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hybrid voice polisher</title>
<style>
{font_face}
{CSS}
</style>
</head>
<body>
<div class="page">
  <header>
    <div class="eyebrow">HYBRID VOICE POLISHER · WINDOWS BETA ARCHITECTURE</div>
    <h1>Audio stays local. Text gets polished.</h1>
    <div class="pipeline">{steps_html}</div>
  </header>

  <main class="main">
    <section class="bench">
      <div class="bench-head">
        <span>GMI-hosted model</span>
        <span class="num">Passed</span>
        <span class="num">Median</span>
        <span class="num">Cost ({n_cases} cases)</span>
      </div>
      {rows_html}
      <p class="bench-note">{note}</p>
    </section>

    <aside class="verbatim">
      <div class="eyebrow-sm">{eyebrow_live}</div>
      <div class="vb-label">Input transcript</div>
      <p class="raw">{live_raw}</p>
      <div class="turn">{arrow_big}<span class="hair"></span></div>
      <div class="vb-label">Polished</div>
      <p class="polished">{live_out}</p>
      <div class="mini">
        <div class="vb-label">Self-correction</div>
        <p class="mini-raw">{self_raw}</p>
        {arrow_small}
        <p class="mini-pol">{self_out}</p>
      </div>
    </aside>
  </main>

  <footer class="foot">
    <span>{foot_left}</span>
    <span>{foot_right}</span>
  </footer>
</div>
</body>
</html>
"""


def launch_browser(p):
    try:
        return p.chromium.launch(headless=True), "playwright-bundled chromium"
    except Exception as exc:  # fall back to an installed Chrome or Edge
        errors = str(exc).splitlines()[0] if str(exc) else "unknown error"
    for candidate in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
    ):
        if candidate.is_file():
            return p.chromium.launch(executable_path=str(candidate), headless=True), str(candidate)
    fail(f"no usable Chromium ({errors})")


def render(doc, out_png):
    with tempfile.TemporaryDirectory(prefix="polisher-card-") as tmp:
        src = Path(tmp) / "card.html"
        src.write_text(doc, encoding="utf-8")
        with sync_playwright() as p:
            browser, used = launch_browser(p)
            try:
                page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
                page.goto(src.as_uri(), wait_until="load")
                page.evaluate("() => document.fonts.ready.then(() => true)")
                page.wait_for_timeout(150)
                faces = page.evaluate(
                    "() => Array.from(document.fonts).map((f) => f.family + ' ' + f.weight + ' ' + f.status)"
                )
                issues = page.evaluate(QA_JS)
                text_all = page.evaluate("() => document.body.innerText")
                page.screenshot(path=str(out_png), type="png")
            finally:
                browser.close()
    return used, faces, issues, text_all


def png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        fail(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


def main():
    d = load_data()
    font_face = ""
    if FONT_FILE.is_file():
        font_face = (
            '@font-face { font-family: "CardInter"; src: url("' + FONT_FILE.as_uri()
            + '") format("woff2"); font-weight: 100 900; font-style: normal; font-display: block; }'
        )
    else:
        print(f"WARNING: font not found at {FONT_FILE}; falling back to system UI font")

    doc = build_html(d, font_face)
    used, faces, issues, text_all = render(doc, OUT_PNG)

    lowered = text_all.lower()
    banned = [w for w in ("sponsor", "@gmi", "#gmi", "\u2014", "\u2013") if w in lowered]
    width, height = png_size(OUT_PNG)

    print(f"wrote {OUT_PNG}")
    print(f"size: {width}x{height} px, {OUT_PNG.stat().st_size} bytes")
    print(f"browser: {used}")
    print(f"fonts: {faces}")
    print(f"layout issues: {issues if issues else 'none'}")
    print(f"banned strings: {banned if banned else 'none'}")
    print("data used:")
    print(json.dumps({k: v for k, v in d.items() if k != "bench"}, ensure_ascii=False, indent=1))
    for row in d["bench"]:
        print(json.dumps(row, ensure_ascii=False))
    if (width, height) != (1920, 1080) or issues or banned:
        sys.exit(1)


if __name__ == "__main__":
    main()
