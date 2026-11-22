#!/usr/bin/env python3
"""Render the 60-second Day 3 hybrid voice polisher video (1920x1080, 30 fps).

Every number, model name, latency, cost, and transcript is read at run time from:
  - benchmark_results.json              (pass counts, medians, costs, config, fail-closed modes)
  - outputs/run_20261122T050023Z.jsonl  (transform outputs; cross-checked against the JSON)
  - test_cases.json                     (raw transcripts and case count)
  - dictionary.json                     (term and replacement counts)

Pipeline:
  1. video/build/voice_polisher_scene.html holds the editorial layer. window.renderFrame(t) draws time t.
  2. Playwright Chromium renders the layer on a transparent background to
     video/build/layer_frames/fNNNNN.png (parallel workers).
  3. ffmpeg composites the layer over warm paper and four Veo 3.1 clips placed at fixed rectangles
     (scaled only, no overlays, no grading). Audio: trimmed to 0-60 s, 0.35 s fade-in,
     2.4 s fade-out from 57.6 s, then a measured gain to -15 LUFS with a true-peak limiter (see the audio section).
  4. ffprobe and ebur128 verification, then 9 QA frames in video/qa_frames/.

Usage, from the project root:
  py -3 video/render_video.py [--workers 4] [--crf 20] [--skip-layer]
"""
import argparse
import array
import concurrent.futures
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

VIDEO = Path(__file__).resolve().parent
ROOT = VIDEO.parent
RAW = VIDEO / "raw"
BUILD = VIDEO / "build"
LAYER_DIR = BUILD / "layer_frames"
HTML_PATH = BUILD / "voice_polisher_scene.html"
OUT_MP4 = VIDEO / "voice-polisher-60s.mp4"
QA_DIR = VIDEO / "qa_frames"
MUSIC = RAW / "progressive_house.mp3"
BENCH = ROOT / "benchmark_results.json"
RUN_LOG = ROOT / "outputs" / "run_20261122T050023Z.jsonl"
TEST_CASES = ROOT / "test_cases.json"
DICTIONARY = ROOT / "dictionary.json"

W, H, FPS = 1920, 1080, 30
DURATION = 60
TOTAL_FRAMES = DURATION * FPS
QA_TIMES = [1.5, 5.0, 9.5, 14.5, 21.5, 32.0, 43.5, 54.5, 58.5]
TRANSFORM_IDS = ["c01", "c02", "c06", "c11"]
GEMINI = "google/gemini-3.8-flash"

# Veo clips: source trim (start, end) in seconds, timeline start (s), fixed rectangle (x, y, w, h).
# Rectangles are even-sized for yuv420p. Clip 1 and clip 4 share the right-hand 62 percent column.
VEO = [
    {"file": "clip1_capture.mp4", "trim": (0.0, 5.8), "at": 0.0, "rect": (760, 230, 1040, 586)},
    {"file": "clip2_prism.mp4", "trim": (0.6, 7.1), "at": 11.5, "rect": (320, 110, 1280, 720)},
    {"file": "clip3_zero_dial.mp4", "trim": (0.0, 8.0), "at": 28.0, "rect": (120, 250, 1040, 586)},
    {"file": "clip4_release.mp4", "trim": (0.0, 5.0), "at": 52.0, "rect": (760, 230, 1040, 586)},
]

AUDIO_PREP = ("atrim=start=0:end=60,asetpts=PTS-STARTPTS,aresample=48000,"
              "afade=t=in:st=0:d=0.35,afade=t=out:st=57.6:d=2.4")


def find_tool(name, env_var):
    path = os.environ.get(env_var) or shutil.which(name)
    if not path:
        raise SystemExit(f"{name} not found on PATH. Set {env_var} to its full path.")
    return path


def find_chromium():
    env = os.environ.get("CHROMIUM_PATH")
    if env and Path(env).is_file():
        return env
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    pw = local / "ms-playwright"
    cands = sorted(pw.glob("chromium_headless_shell-*/chrome-headless-shell-win64/chrome-headless-shell.exe"))
    cands += sorted(pw.glob("chromium-*/chrome-win64/chrome.exe"))
    for c in cands:
        if c.is_file():
            return str(c)
    raise SystemExit("No Playwright Chromium found. Run: py -3 -m playwright install chromium")


def short_error(message, cfg):
    msg = message.replace("error: ", "").strip()
    if re.search(r"timed out after [0-9.]+ s", msg):
        return f"timed out ({cfg['timeout_s']:g} s configured)"
    if "cut off at max_tokens" in msg:
        return f"cut off at {cfg['max_tokens']}-token cap"
    return msg[:48]


def load_data():
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    tc = json.loads(TEST_CASES.read_text(encoding="utf-8"))
    dic = json.loads(DICTIONARY.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in RUN_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    cfg = bench["config"]
    models = bench["models"]
    sweep = [r for r in rows if r["phase"] == "sweep"]
    tc_by_id = {c["id"]: c for c in tc["cases"]}

    # Cross-check benchmark_results.json against the sweep rows of the raw run log.
    for mid, m in models.items():
        srows = [r for r in sweep if r["model"] == mid]
        lat = [x for r in srows for x in r["latencies_s"]]
        chk = {
            "model": mid,
            "passed_json": m["cases_passed"], "passed_log": sum(1 for r in srows if r["ok"]),
            "list_json": m["cost_usd"]["list"], "list_log": round(sum(r["list_usd"] for r in srows), 6),
            "after_json": m["cost_usd"]["after_discount"],
            "after_log": round(sum(r["effective_usd"] for r in srows), 6),
            "median_json": m["latency_per_request_s"]["median"],
            "median_from_log": round(statistics.median(lat), 3),
            "n_json": m["latency_per_request_s"]["n"], "n_from_log": len(lat),
        }
        print("cross-check:", json.dumps(chk))
        if (chk["passed_json"] != chk["passed_log"]
                or abs(chk["list_json"] - chk["list_log"]) > 1e-9
                or abs(chk["after_json"] - chk["after_log"]) > 1e-9):
            raise SystemExit(f"benchmark_results.json disagrees with the run log: {chk}")

    model_rows = []
    for mid, m in models.items():
        fails = [c for c in bench["cases"] if c["model"] == mid and not c["ok"]]
        if fails:
            reasons = []
            for c in fails:
                for f in c["failures"]:
                    s = short_error(f, cfg)
                    if s not in reasons:
                        reasons.append(s)
            fail_lines = ["failed: " + ", ".join(c["id"] for c in fails), "; ".join(reasons)]
        else:
            fail_lines = ["no failed cases", ""]
        model_rows.append({
            "id": mid,
            "short": mid.split("/")[-1],
            "passed_str": f"{m['cases_passed']}/{m['cases_total']}",
            "median": m["latency_per_request_s"]["median"],
            "median_str": f"{m['latency_per_request_s']['median']:g} s",
            "cost_list_str": f"list ${m['cost_usd']['list']:.6f}",
            "cost_eff_str": f"after ${m['cost_usd']['after_discount']:.6f}",
            "fail_lines": fail_lines,
            "primary": mid == GEMINI,
        })
    max_median = max(r["median"] for r in model_rows)

    sweep_by = {(r["model"], r["case_id"]): r for r in sweep if r["run_index"] == 1}
    examples = {e["case_id"]: e for e in bench.get("examples", [])}
    transforms = []
    for cid in TRANSFORM_IDS:
        row = sweep_by[(GEMINI, cid)]
        if not row["ok"]:
            raise SystemExit(f"{cid} did not pass in the run log")
        if cid in examples:
            if examples[cid]["after"] != row["output"]:
                raise SystemExit(f"{cid} output differs from benchmark_results.json examples")
            if examples[cid]["before"] != tc_by_id[cid]["transcript"]:
                raise SystemExit(f"{cid} transcript differs from test_cases.json")
        transforms.append({
            "id": cid,
            "category": row["category"].replace("_", " "),
            "before": tc_by_id[cid]["transcript"],
            "after": row["output"],
        })

    gem = models[GEMINI]
    fc = bench["fail_closed"]
    fc_pass = sum(1 for m in fc["modes"] if m["passed"])
    fc_total = len(fc["modes"])
    if fc["score"] != f"{fc_pass} of {fc_total}":
        raise SystemExit(f"fail_closed score {fc['score']!r} does not match its modes")

    temp = f"{cfg['temperature']:.1f}"
    det = bench["determinism"]
    gem_short = GEMINI.split("/")[-1]
    return {
        "run_id": bench["run_id"],
        "run_log": "outputs/" + RUN_LOG.name,
        "model_id": GEMINI,
        "model_short": gem_short,
        "temperature": temp,
        "n_cases": len(tc["cases"]),
        "n_models": len(models),
        "dict_terms": len(dic["terms"]),
        "dict_repl": len(dic["replacements"]),
        "transforms": transforms,
        "models": model_rows,
        "max_median": max_median,
        "specs": [
            ["temperature", temp],
            ["max_tokens", str(cfg["max_tokens"])],
            ["chunk_words", str(cfg["chunk_words"])],
            ["timeout_s", f"{cfg['timeout_s']:g} s"],
            ["max_split_depth", str(cfg["max_split_depth"])],
            ["fail_closed", fc["score"]],
        ],
        "steps": [
            ["01", "Audio stays local. The recognizer turns speech into text."],
            ["02", f"Text is split into chunks of at most {cfg['chunk_words']} words."],
            ["03", f"Only text and matched dictionary hints go out over HTTPS, at temperature {temp}."],
            ["04", "Any failure raises PolisherError. Raw ASR text is never returned."],
        ],
        "hero1": f"{gem['cases_passed']}/{gem['cases_total']}",
        "hero1_cap": f"{gem_short} cases passed in the sweep.",
        "hero2": f"{fc_pass}/{fc_total}",
        "hero2_cap": "fail-closed modes ended in an error, not a guess.",
        "footnote": (f"Determinism at temperature {temp}: {det['score']} "
                     f"({det['runs_per_case']} runs per case). "
                     f"Source: benchmark_results.json, run {bench['run_id']}."),
        "breakdown_sub": (f"Same transcripts and checks on every model. Temperature {temp}, "
                          f"{cfg['max_tokens']}-token cap, {cfg['timeout_s']:g} s timeout."),
        "breakdown_src": (f"Counts from test_cases.json ({len(tc['cases'])} cases) and "
                          f"benchmark_results.json ({len(models)} models)."),
        "end_sub": f"Parakeet TDT 0.6B v2 on Windows. GMI Inference Engine cleanup at temperature {temp}.",
        "_checks_note": "sweep-phase pass counts, list and after-discount costs matched the JSON",
    }


# ---------------------------------------------------------------------------
# Editorial layer (HTML). Every number arrives through the __DATA__ placeholder.
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>voice polisher layer</title>
<style>
html,body{margin:0;padding:0;background:transparent;overflow:hidden}
#stage{position:relative;width:1920px;height:1080px;overflow:hidden;background:transparent;font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;color:#141414}
.abs{position:absolute;margin:0;padding:0}
.nw{white-space:nowrap}
.t-mono{font-family:Consolas,"Cascadia Mono","Courier New",monospace}
.t-mono17{font-family:Consolas,"Cascadia Mono","Courier New",monospace;font-size:17px;line-height:1.3;color:#141414;white-space:nowrap}
.t-mono18{font-family:Consolas,"Cascadia Mono","Courier New",monospace;font-size:18px;line-height:1.3;color:#141414;white-space:nowrap}
.t-mono20{font-family:Consolas,"Cascadia Mono","Courier New",monospace;font-size:20px;line-height:1.3;color:#141414;white-space:nowrap}
.t-mono22{font-family:Consolas,"Cascadia Mono","Courier New",monospace;font-size:22px;line-height:1.3;color:#141414;white-space:nowrap}
.t-mono26{font-family:Consolas,"Cascadia Mono","Courier New",monospace;font-size:26px;line-height:1.2;color:#141414;white-space:nowrap}
.t-kicker{font-family:Consolas,"Cascadia Mono","Courier New",monospace;font-size:17px;line-height:1.2;letter-spacing:0.16em;color:#8A857C;white-space:nowrap}
.t-head{font-size:56px;line-height:1.1;font-weight:600;letter-spacing:-0.015em;color:#141414;white-space:nowrap}
.t-title{font-size:96px;line-height:104px;font-weight:600;letter-spacing:-0.02em;color:#141414}
.t-end{font-size:128px;line-height:128px;font-weight:600;letter-spacing:-0.025em;color:#141414}
.t-big{font-size:120px;line-height:1;font-weight:600;letter-spacing:-0.03em;color:#141414;white-space:nowrap}
.t-hero{font-size:112px;line-height:1;font-weight:600;letter-spacing:-0.03em;color:#D94E1F;white-space:nowrap}
.t-body{font-size:26px;line-height:1.38;color:#57534E}
.t-body22{font-size:22px;line-height:1.35;color:#57534E}
.t-body24{font-size:24px;line-height:1.35;color:#57534E;white-space:nowrap}
.t-body26{font-size:26px;line-height:1.35;color:#57534E;white-space:nowrap}
.t-body34{font-size:34px;line-height:1.35;color:#57534E}
.t-body36{font-size:36px;line-height:1.3;font-weight:600;color:#141414;white-space:nowrap}
.t-caption{font-size:24px;line-height:1.3;color:#57534E;white-space:nowrap}
.t-node{font-size:40px;line-height:1.1;font-weight:600;letter-spacing:-0.01em;color:#141414;white-space:nowrap}
.t-name{font-size:36px;line-height:1.2;font-weight:600;color:#141414;white-space:nowrap}
.t-num{font-size:56px;line-height:1.1;font-weight:600;letter-spacing:-0.02em;color:#141414;white-space:nowrap}
.t-legend{font-size:34px;line-height:1.1;font-weight:600;color:#141414;white-space:nowrap}
.t-arrow{font-size:36px;line-height:1;color:#8A857C;text-align:center}
.t-raw{font-size:42px;line-height:1.25;color:#57534E}
.t-pol{font-size:56px;line-height:1.2;font-weight:500;color:#141414}
.t-spec{font-family:Consolas,"Cascadia Mono","Courier New",monospace;font-size:40px;line-height:1.2;color:#141414;white-space:nowrap}
.t-step{font-size:30px;line-height:1.3;color:#141414}
.acc{color:#D94E1F}
.g2{color:#8A857C}
</style></head>
<body><div id="stage"></div>
<script>
const D = __DATA__;
const stage = document.getElementById('stage');
const INK = '#141414', G1 = '#57534E', G2 = '#8A857C', G3 = '#DCD7CE', ACC = '#D94E1F';
const NODES = [];
function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
function node(o) {
  const el = document.createElement('div');
  el.className = 'abs' + (o.cls ? ' ' + o.cls : '');
  el.style.left = (o.x || 0) + 'px';
  el.style.top = (o.y || 0) + 'px';
  if (o.w != null) el.style.width = o.w + 'px';
  if (o.h != null) el.style.height = o.h + 'px';
  if (o.style) Object.assign(el.style, o.style);
  if (o.html != null) el.innerHTML = o.html;
  el.style.display = 'none';
  stage.appendChild(el);
  const n = {
    el: el,
    s: o.s != null ? o.s : 0,
    e: o.e != null ? o.e : 60,
    dur: o.dur != null ? o.dur : 0.45,
    dy: o.dy != null ? o.dy : 12,
    fade: o.fade !== false,
    grow: o.grow || null
  };
  NODES.push(n);
  return n;
}
function T(x, y, w, html, o) {
  return node(Object.assign({cls: 't-body'}, o || {}, {x: x, y: y, w: w, html: html}));
}
function R(x, y, w, h, color, o) {
  return node(Object.assign({dy: 0, fade: false, style: {background: color}}, o || {}, {x: x, y: y, w: w, h: h}));
}
window.renderFrame = function (t) {
  for (const n of NODES) {
    if (t < n.s - 1e-6 || t >= n.e - 1e-6) { n.el.style.display = 'none'; continue; }
    n.el.style.display = 'block';
    const p = n.dur > 0 ? clamp((t - n.s) / n.dur, 0, 1) : 1;
    const e = 1 - Math.pow(1 - p, 3);
    n.el.style.opacity = n.fade ? e.toFixed(4) : '1';
    n.el.style.transform = n.dy ? 'translate3d(0,' + ((1 - e) * n.dy).toFixed(3) + 'px,0)' : 'none';
    if (n.grow) n.el.style.width = (n.grow.full * e).toFixed(3) + 'px';
  }
};

function buildScene() {
  // Persistent header
  T(120, 40, 900, 'DAY 3 · HYBRID VOICE POLISHER', {cls: 't-kicker nw', s: 0, e: 57, dy: 0, fade: false});
  R(120, 92, 1680, 1, G3, {s: 0, e: 57});
  function hdr(label, s, e) {
    T(1000, 40, 800, label, {cls: 't-kicker nw', s: s, e: e, dy: 0, fade: false, style: {textAlign: 'right'}});
  }

  // ACT 1 (0-8 s): title, 38/62 split, Veo clip 1 on the right
  hdr('01 / 06 · INTRO', 0, 8);
  R(730, 0, 1, 1080, G3, {s: 0, e: 8});
  T(120, 250, 620, '<div>Audio stays</div><div>local<span class="acc">.</span></div><div>Text gets</div><div>polished<span class="acc">.</span></div>',
    {cls: 't-title', s: 0, e: 8, dur: 0.7, dy: 18});
  T(120, 700, 560, 'Speech is transcribed on this PC. Only text chunks and dictionary hints go to GMI Inference Engine over HTTPS.',
    {cls: 't-body', s: 0.7, e: 8, dur: 0.6});
  T(760, 842, 1040, '<span class="t-mono g2">CAPTURE</span>&nbsp;&nbsp;Mic audio is transcribed on this PC.',
    {cls: 't-caption', s: 0.5, e: 8});

  // ACT 2 (8-18 s): pipeline 8-11.5 s, then Veo clip 2 centred with a three-step legend
  hdr('02 / 06 · PIPELINE', 8, 18);
  T(120, 150, 1680, 'How one dictation moves.', {cls: 't-head', s: 8, e: 11.5});
  T(120, 232, 1680, 'Audio stays on this PC. Only text chunks leave it.', {cls: 't-body', s: 8, e: 11.5});
  const pipe = [
    {k: '01 · LOCAL', t: 'Microphone', sub: 'Captured on Windows.', col: INK},
    {k: '02 · LOCAL', t: 'Parakeet ASR', sub: 'TDT 0.6B v2 via sherpa-onnx, on device.', col: INK},
    {k: '03 · LOCAL', t: 'dictionary.json', sub: D.dict_terms + ' terms · ' + D.dict_repl + ' replacement', col: INK},
    {k: '04 · CLOUD · HTTPS', t: 'GMI Cloud', sub: 'Temp 0 · text chunks only', col: ACC}
  ];
  pipe.forEach(function (p, i) {
    const x = 120 + i * 435, s = 8 + 0.25 * i;
    R(x, 390, 375, 2, p.col, {s: s, e: 11.5});
    T(x, 412, 375, p.k, {cls: 't-kicker nw', s: s, e: 11.5, dy: 0, fade: false});
    T(x, 446, 375, p.t, {cls: 't-node', s: s, e: 11.5, dur: 0.4});
    T(x, 500, 375, p.sub, {cls: 't-body22', s: s, e: 11.5, dur: 0.4});
    if (i < 3) T(x + 375, 452, 60, '→', {cls: 't-arrow', s: s + 0.15, e: 11.5, dy: 0});
  });
  const legend = [
    ['01', 'Speak', 'Parakeet transcribes on this PC. No GPU or CUDA needed.'],
    ['02', 'Clean', 'Text chunks and dictionary hints go out over HTTPS at temperature ' + D.temperature + '.'],
    ['03', 'Return', 'Polished text comes back. Failures raise; raw ASR text is never returned.']
  ];
  legend.forEach(function (L, i) {
    const x = 320 + i * 440;
    T(x, 868, 400, L[0], {cls: 't-mono22 acc', s: 11.5, e: 18, dy: 0});
    T(x, 898, 400, L[1], {cls: 't-legend', s: 11.5, e: 18});
    T(x, 946, 400, L[2], {cls: 't-body22', s: 11.5, e: 18});
  });

  // ACT 3 (18-28 s): four real transforms, 2.5 s each, from the run log
  hdr('03 / 06 · REAL TRANSFORMS', 18, 28);
  T(120, 150, 1680, 'Four real transforms from the run log.', {cls: 't-head', s: 18, e: 28});
  D.transforms.forEach(function (tr, k) {
    const cs = 18 + 2.5 * k, ce = cs + 2.5;
    T(120, 236, 1680, '<span class="acc">' + tr.id + '</span>&nbsp;&nbsp;<span class="g2">' + tr.category + '</span>',
      {cls: 't-mono22', s: cs, e: ce, dur: 0.3, dy: 10});
    T(120, 300, 1000, 'INPUT TRANSCRIPT', {cls: 't-kicker nw', s: cs, e: ce, dur: 0.3, dy: 10});
    T(120, 334, 1680, tr.before, {cls: 't-raw', s: cs + 0.1, e: ce, dur: 0.4, dy: 12});
    R(120, 500, 1680, 1, G3, {s: cs, e: ce});
    T(120, 536, 1000, 'POLISHED · ' + D.model_short.toUpperCase(), {cls: 't-kicker nw', s: cs, e: ce, dur: 0.3, dy: 10});
    T(120, 572, 1680, tr.after, {cls: 't-pol', s: cs + 0.25, e: ce, dur: 0.4, dy: 12});
  });
  for (let k = 0; k < 4; k++) {
    R(120 + k * 426, 930, 400, 4, G3, {s: 18, e: 28});
    R(120 + k * 426, 930, 0, 4, INK, {s: 18 + 2.5 * k, e: 28, dur: 2.5, fade: false, dy: 0, grow: {full: 400}});
  }
  T(120, 966, 1680, D.run_log + ' · ' + D.model_id + ' · temperature ' + D.temperature,
    {cls: 't-mono18 g2', s: 18, e: 28});

  // ACT 4 (28-40 s): Veo clip 3 with guardrail specs (28-36), breakdown card (36-40)
  hdr('04 / 06 · GUARDRAILS', 28, 36);
  hdr('04 / 06 · BREAKDOWN', 36, 40);
  T(120, 150, 1700, 'Fail closed. Never guess.', {cls: 't-head', s: 28, e: 36});
  T(120, 872, 1040, 'Every failure raises PolisherError. There is no local fallback.', {cls: 't-caption', s: 28, e: 36});
  T(1200, 250, 600, 'GUARDRAILS · BENCHMARK_RESULTS.JSON', {cls: 't-kicker nw', s: 28, e: 36});
  D.specs.forEach(function (sp, i) {
    const y = 300 + 96 * i, s = 28.2 + 0.3 * i;
    T(1200, y + 30, 300, sp[0], {cls: 't-mono20 g2', s: s, e: 36, dur: 0.35, dy: 8});
    T(1500, y + 22, 300, sp[1], {cls: 't-spec', s: s, e: 36, dur: 0.35, dy: 8});
    R(1200, y + 92, 600, 1, G3, {s: s, e: 36});
  });
  T(120, 300, 1000, 'BREAKDOWN', {cls: 't-kicker acc nw', s: 36, e: 40});
  T(120, 340, 1680, D.n_cases + ' cases. ' + D.n_models + ' models<span class="acc">.</span>',
    {cls: 't-big', s: 36, e: 40, dur: 0.5});
  T(120, 520, 1500, D.breakdown_sub, {cls: 't-body34', s: 36.3, e: 40});
  T(120, 650, 1680, D.breakdown_src, {cls: 't-mono20 g2', s: 36.6, e: 40});

  // ACT 5 (40-52 s): scoreboard with bars, medians, costs, and hero numbers
  hdr('05 / 06 · SCOREBOARD', 40, 52);
  T(120, 110, 1680, 'Same ' + D.n_cases + ' cases. ' + D.n_models + ' models.', {cls: 't-head', s: 40, e: 52});
  T(120, 180, 1680, 'Pass count, median latency per request, and run cost, from benchmark_results.json.',
    {cls: 't-body24', s: 40, e: 52});
  T(120, 218, 1680, 'GMI-hosted models only, not the whole market.', {cls: 't-body24', s: 40, e: 52});
  T(120, 270, 380, 'MODEL', {cls: 't-kicker nw', s: 40, e: 52, dy: 0, fade: false});
  T(520, 270, 340, 'PASSED', {cls: 't-kicker nw', s: 40, e: 52, dy: 0, fade: false});
  T(900, 270, 480, 'MEDIAN LATENCY PER REQUEST', {cls: 't-kicker nw', s: 40, e: 52, dy: 0, fade: false});
  T(1560, 270, 260, 'RUN COST', {cls: 't-kicker nw', s: 40, e: 52, dy: 0, fade: false});
  D.models.forEach(function (m, i) {
    const y0 = 310 + 160 * i, s0 = 40 + 0.2 * i;
    T(120, y0, 380, m.short, {cls: 't-name', s: s0, e: 52});
    T(120, y0 + 50, 380, m.id, {cls: 't-mono17 g2', s: s0, e: 52});
    T(520, y0 - 6, 340, m.passed_str, {cls: 't-num', s: s0, e: 52});
    T(520, y0 + 60, 360, m.fail_lines[0], {cls: 't-mono17 g2', s: s0, e: 52});
    T(520, y0 + 84, 360, m.fail_lines[1], {cls: 't-mono17 g2', s: s0, e: 52});
    R(900, y0 + 24, 480, 14, G3, {s: s0, e: 52});
    const full = 480 * m.median / D.max_median;
    R(900, y0 + 24, 0, 14, m.primary ? INK : G2,
      {s: s0 + 0.6, e: 52, dur: 1.0, fade: false, dy: 0, grow: {full: full}});
    T(900 + full + 16, y0 + 16, 200, m.median_str, {cls: 't-mono26', s: s0 + 1.0, e: 52, dy: 0});
    T(1560, y0 + 4, 260, m.cost_list_str, {cls: 't-mono22', s: s0, e: 52});
    T(1560, y0 + 38, 260, m.cost_eff_str, {cls: 't-mono20 g2', s: s0, e: 52});
    R(120, y0 + 124, 1680, 1, G3, {s: 40, e: 52});
  });
  R(120, 800, 1680, 2, INK, {s: 40, e: 52});
  T(120, 822, 800, D.hero1, {cls: 't-hero', s: 40.8, e: 52, dur: 0.6});
  T(120, 948, 800, D.hero1_cap, {cls: 't-body24', s: 40.8, e: 52});
  T(960, 822, 800, D.hero2, {cls: 't-hero', s: 41.2, e: 52, dur: 0.6});
  T(960, 948, 800, D.hero2_cap, {cls: 't-body24', s: 41.2, e: 52});
  T(120, 1010, 1680, D.footnote, {cls: 't-mono18 g2', s: 41.5, e: 52});

  // ACT 6 (52-60 s): Veo clip 4 with workflow copy (52-57), end card (57-60)
  hdr('06 / 06 · WORKFLOW', 52, 57);
  T(120, 250, 520, 'HOW IT RUNS', {cls: 't-kicker nw', s: 52, e: 57, dy: 0, fade: false});
  D.steps.forEach(function (st, i) {
    const y = 300 + 130 * i, s = 52.2 + 0.3 * i;
    T(120, y + 4, 80, st[0], {cls: 't-mono22 acc', s: s, e: 57, dy: 0});
    T(180, y, 520, st[1], {cls: 't-step', s: s, e: 57, dur: 0.4, dy: 10});
  });
  T(760, 842, 1040, '<span class="t-mono g2">RELEASE</span>&nbsp;&nbsp;Local capture, cloud cleanup, text only.',
    {cls: 't-caption', s: 52.4, e: 57});
  T(120, 250, 1500, '<div>Audio stays</div><div>local<span class="acc">.</span></div><div>Text gets</div><div>polished<span class="acc">.</span></div>',
    {cls: 't-end', s: 57, e: 60, dur: 0.5, dy: 14});
  T(120, 800, 1600, 'Day 3 · Hybrid voice polisher', {cls: 't-body36', s: 57.4, e: 60});
  T(120, 852, 1600, D.end_sub, {cls: 't-body26', s: 57.6, e: 60});
}

buildScene();
const fontsReady = (document.fonts && document.fonts.ready) ? document.fonts.ready : Promise.resolve();
fontsReady.then(function () { window.renderFrame(0); window.__ready = true; });
</script></body></html>
"""


def build_html(data):
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA__", payload)


# ---------------------------------------------------------------------------
# Layer rendering (Playwright, parallel workers, transparent PNG frames)
# ---------------------------------------------------------------------------
def _render_range(job):
    start, end, url, chromium = job
    from playwright.sync_api import sync_playwright

    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chromium,
            headless=True,
            args=["--hide-scrollbars", "--force-device-scale-factor=1", "--force-color-profile=srgb"],
        )
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda msg: errors.append("console: " + msg.text) if msg.type == "error" else None)
        page.goto(url, wait_until="load")
        page.wait_for_function("window.__ready === true", timeout=30000)
        if errors:
            browser.close()
            raise RuntimeError("scene errors: " + " | ".join(errors))
        for i in range(start, end):
            page.evaluate("t => window.renderFrame(t)", i / FPS)
            page.screenshot(path=str(LAYER_DIR / f"f{i:05d}.png"), omit_background=True, type="png")
        browser.close()
    return end - start


def render_layer(workers):
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    chromium = find_chromium()
    url = HTML_PATH.resolve().as_uri()
    per = -(-TOTAL_FRAMES // workers)
    jobs = [(s, min(TOTAL_FRAMES, s + per), url, chromium) for s in range(0, TOTAL_FRAMES, per)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        done = sum(ex.map(_render_range, jobs))
    if done != TOTAL_FRAMES:
        raise SystemExit(f"rendered {done} frames, expected {TOTAL_FRAMES}")
    return done


# ---------------------------------------------------------------------------
# Audio: trim and fades, then one measured linear gain to -15 LUFS and a
# true-peak limiter. Loudnorm was tried first. Its dynamic mode landed at
# -13.1 LUFS, and its linear fallback at -14.5 LUFS with a 0.0 dBTP decoded
# peak, so the delivered chain uses the gain solved below instead.
# AAC encoding adds about 1 dB of overshoot, so the limiter ceiling sits at
# -3.0 dBFS and the decoded file measures about -1.9 dBTP.
# ---------------------------------------------------------------------------
TARGET_I = -15.0
TP_CEILING_DB = -3.0


def chain_with_gain(gain_db):
    lin = 10 ** (TP_CEILING_DB / 20)
    return f"{AUDIO_PREP},volume={gain_db:.3f}dB,alimiter=limit={lin:.4f}:attack=2:release=60:level=0"


def measure_integrated(ffmpeg, af):
    p = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(MUSIC), "-vn", "-af", af + ",ebur128=peak=true",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    s = p.stderr[p.stderr.rfind("Summary:"):]
    m = re.search(r"\bI:\s+(-?[0-9.]+)\s+LUFS", s)
    if p.returncode != 0 or not m:
        raise SystemExit("integrated loudness measurement failed:\n" + p.stderr[-2000:])
    return float(m.group(1))


def solve_gain(ffmpeg):
    gain = TARGET_I - measure_integrated(ffmpeg, AUDIO_PREP)
    for _ in range(6):
        i_val = measure_integrated(ffmpeg, chain_with_gain(gain))
        if abs(i_val - TARGET_I) < 0.051:
            return round(gain, 3), i_val
        gain += TARGET_I - i_val
    raise SystemExit("gain solver did not converge")

def energy_profile(ffmpeg):
    """Per-second RMS in dBFS of the trimmed music, for checking section placement."""
    p = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-t", "60", "-i", str(MUSIC), "-vn",
         "-ac", "1", "-ar", "8000", "-f", "s16le", "-"], capture_output=True)
    if p.returncode != 0:
        raise SystemExit("energy profile decode failed")
    samples = array.array("h")
    samples.frombytes(p.stdout)
    sr = 8000
    out = []
    for sec in range(DURATION):
        seg = samples[sec * sr:(sec + 1) * sr]
        if not seg:
            break
        rms = math.sqrt(sum(x * x for x in seg) / len(seg)) / 32768.0
        out.append(round(20 * math.log10(rms + 1e-12), 1))
    return out


# ---------------------------------------------------------------------------
# Compositing and encode
# ---------------------------------------------------------------------------
def build_filter_complex(gain_db):
    parts = [
        "color=c=0xF5F3EE:s=1920x1080:r=30:d=60,format=rgb24,"
        "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[base]",
        "[0:v]format=rgba,scale=out_color_matrix=bt709:out_range=tv,format=yuva420p,"
        "setpts=PTS-STARTPTS[lay]",
    ]
    cur = "base"
    for idx, v in enumerate(VEO, start=1):
        ts, te = v["trim"]
        x, y, w, h = v["rect"]
        at = v["at"]
        end = round(at + (te - ts), 3)
        parts.append(f"[{idx}:v]trim=start={ts}:end={te},setpts=PTS-STARTPTS,fps={FPS},"
                     f"scale={w}:{h}:flags=lanczos,format=yuv420p,setpts=PTS+{at}/TB[v{idx}]")
        parts.append(f"[{cur}][v{idx}]overlay=x={x}:y={y}:enable='gte(t,{at})*lt(t,{end})':"
                     f"eof_action=pass[o{idx}]")
        cur = f"o{idx}"
    parts.append(f"[{cur}][lay]overlay=x=0:y=0:format=auto,format=yuv420p[vout]")
    parts.append(f"[{len(VEO) + 1}:a]{chain_with_gain(gain_db)}[aout]")
    return ";".join(parts)


def run(cmd, label):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise SystemExit(f"{label} failed ({p.returncode}):\n{p.stderr[-3000:]}")
    return p


def probe(ffprobe, path):
    p = run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], "ffprobe")
    return json.loads(p.stdout)


def atom_order(path):
    size = path.stat().st_size
    order = []
    with open(path, "rb") as f:
        off = 0
        while off < size and len(order) < 50:
            f.seek(off)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            n = int.from_bytes(hdr[:4], "big")
            typ = hdr[4:8].decode("latin-1")
            if n == 1:
                n = int.from_bytes(f.read(8), "big")
            elif n == 0:
                n = size - off
            order.append(typ)
            if n < 8:
                break
            off += n
    return order


def ebur_check(ffmpeg):
    p = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(OUT_MP4), "-map", "0:a:0",
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    s = p.stderr[p.stderr.rfind("Summary:"):]
    i_val = re.search(r"\bI:\s+(-?[0-9.]+)\s+LUFS", s)
    lra = re.search(r"LRA:\s+(-?[0-9.]+)\s+LU", s)
    peaks = re.findall(r"Peak:\s+(-?[0-9.]+)\s+dBFS", s)
    if not (i_val and lra and peaks):
        raise SystemExit("ebur128 summary not parsed:\n" + p.stderr[-1500:])
    return {"integrated_lufs": float(i_val.group(1)), "lra": float(lra.group(1)),
            "true_peak_dbtp": float(peaks[-1])}


def verify(ffmpeg, ffprobe, energy):
    info = probe(ffprobe, OUT_MP4)
    streams = info["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    a = next(s for s in streams if s["codec_type"] == "audio")
    fmt = info["format"]
    order = atom_order(OUT_MP4)
    faststart = "moov" in order and "mdat" in order and order.index("moov") < order.index("mdat")
    return {
        "file": str(OUT_MP4),
        "size_bytes": OUT_MP4.stat().st_size,
        "size_mb": round(OUT_MP4.stat().st_size / 1e6, 2),
        "format_duration_s": float(fmt["duration"]),
        "video": {k: v.get(k) for k in ("codec_name", "profile", "pix_fmt", "width", "height", "r_frame_rate",
                                        "avg_frame_rate", "nb_frames", "duration", "start_time",
                                        "color_space", "color_primaries", "color_transfer", "color_range")},
        "audio": {k: a.get(k) for k in ("codec_name", "profile", "sample_rate", "channels", "channel_layout",
                                        "duration", "start_time", "bit_rate")},
        "faststart": faststart,
        "top_level_atoms": order[:6],
        "loudness": ebur_check(ffmpeg),
        "energy_dbfs_per_second": energy,
    }


def extract_qa(ffmpeg):
    QA_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    for t in QA_TIMES:
        out = QA_DIR / f"qa_{t:04.1f}s.png"
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(OUT_MP4),
             "-ss", f"{t}", "-frames:v", "1", str(out)], f"QA frame {t}")
        outs.append(str(out))
    return outs


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Render the 60 s Day 3 voice polisher video.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--skip-layer", action="store_true", help="reuse existing build/layer_frames")
    args = ap.parse_args()

    ffmpeg = find_tool("ffmpeg", "FFMPEG_PATH")
    ffprobe = find_tool("ffprobe", "FFPROBE_PATH")
    data = load_data()
    BUILD.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(build_html(data), encoding="utf-8")
    print("scene html:", HTML_PATH)

    t0 = time.time()
    if args.skip_layer:
        n = len(list(LAYER_DIR.glob("f*.png")))
        if n != TOTAL_FRAMES:
            raise SystemExit(f"--skip-layer needs {TOTAL_FRAMES} frames, found {n}")
    else:
        render_layer(args.workers)
    t1 = time.time()
    print(f"layer frames ready in {t1 - t0:.1f} s")

    gain_db, i_pre = solve_gain(ffmpeg)
    print(f"audio gain {gain_db:+.3f} dB; pre-encode integrated {i_pre} LUFS; limiter ceiling {TP_CEILING_DB} dBFS")
    energy = energy_profile(ffmpeg)

    fc = build_filter_complex(gain_db)
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-y", "-loglevel", "error",
           "-framerate", str(FPS), "-start_number", "0", "-i", str(LAYER_DIR / "f%05d.png")]
    for v in VEO:
        cmd += ["-i", str(RAW / v["file"])]
    cmd += ["-i", str(MUSIC),
            "-filter_complex", fc,
            "-map", "[vout]", "-map", "[aout]",
            "-frames:v", str(TOTAL_FRAMES), "-t", str(DURATION),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", "60", "-bf", "2",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", str(OUT_MP4)]
    run(cmd, "final encode")
    t2 = time.time()
    print(f"encode finished in {t2 - t1:.1f} s; total {t2 - t0:.1f} s")

    report = verify(ffmpeg, ffprobe, energy)
    report["render_seconds_layer"] = round(t1 - t0, 1)
    report["render_seconds_total"] = round(t2 - t0, 1)
    report["checks"] = data["_checks_note"]
    loud = report["loudness"]
    report["targets_met"] = {
        "integrated_within_0.5_of_-15": abs(loud["integrated_lufs"] + 15) <= 0.5,
        "true_peak_at_or_below_-1.5_dbtp": loud["true_peak_dbtp"] <= -1.5,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    qa = extract_qa(ffmpeg)
    print("QA frames:")
    for q in qa:
        print("  " + q)


if __name__ == "__main__":
    main()
