# Cost Log: Hybrid Local-ASR + GMI Cloud Voice Polisher

Paths below are relative to the repository root.

Status: COMPLETE, 2026-11-22. All 45 benchmark rows (36 sweep cases across 3 GMI-hosted models + 9 Gemini repeat runs), 2 live probes, 5 offline fail-closed safety probes, and 5 GMI media generations (4 Veo 3.1 1080p cinema clips + 1 Minimax Music 3.0 soundtrack) completed.

GMI-hosted models only, not the whole market. This log does not compare GMI with any other provider.

Spend basis (reported separately from account-window billing deltas):
- Token-priced list estimate: prompt tokens x prompt rate + completion tokens x completion rate, from `pricing_snapshot.json` (verified against live GMI MCP catalog reads on 2026-11-22).
- Token-priced discounted estimate: list x (1 - `discount_to_user`).
- Media: per-second / per-request list prices confirmed via `gmi___get_model` and `gmi___estimate_generation` before every `gmi___submit_generation` (zero discount on media).

## Sources

| Figure | Source | Status |
|---|---|---|
| Account snapshot, pre-run | Live MCP read `gmi___get_account_summary` on 2026-11-22 | Recorded before Day 3 calls |
| Account snapshot, post-run | Live MCP read `gmi___get_account_summary` on 2026-11-22 after all LLM and media calls settled | Verified: `balance_usd: 0.00`, `promotion_credit_usd: 31.70` |
| LLM model pricing | `pricing_snapshot.json` verified via live `gmi___get_model` (`google/gemini-3.8-flash`) and `gmi___search_models` (`kind="llm"`) on 2026-11-22 | Exact match across all 3 models |
| Media quotes | Live `gmi___estimate_generation` calls before every `gmi___submit_generation` | Approved by user before spend |
| Benchmark rows, statuses, latencies, aggregate token usage, cleaned outputs | `outputs/run_20261122T050023Z.jsonl`, `benchmark_results.json` | 45 rows logged (36 sweep + 9 Gemini repeats) |
| Offline unit tests | `python -m unittest test_polisher.py -v` | 47 of 47 passed |
| Dry-run worst-case ceiling (estimate) | `python benchmark_polisher.py --dry-run` (prompt tokens estimated at 3 chars/token) | `$0.416979` list / `$0.393895` after discount (full suite including retries); `$0.254909` list for sweep alone |

## Account-window snapshot (separate from token-priced run estimates)

| Point | balance_usd | promotion_credit_usd | 7-day LLM token cost (account-wide) | Source |
|---|---|---|---|---|
| Pre-run read, 2026-11-22 | 0.00 | 38.26 | $4.854938 (94 requests, 6,737,375 tokens) | `gmi___get_account_summary` |
| Post-run settled read, 2026-11-22 | 0.00 | 31.70 | $4.866591 (154 requests, 6,768,618 tokens) | `gmi___get_account_summary` |
| **Account-window delta** | **0.00** | **-6.56** | **+$0.011653 (+60 requests, +31,243 tokens)** | **Post minus pre** |

Cash balance and promotion credit are separate pools and are never summed.

Note on account-window vs local run logs: the observed `promotion_credit_usd` drop (`-$6.56`) equals the `$6.55` media spend plus the account-window LLM delta (`+$0.011653`, rounding to `$6.56`). The local JSONL benchmark log (`outputs/run_20261122T050023Z.jsonl`, 55 HTTP attempts, 27,617 tokens, `$0.011109` discounted estimate) plus the logged probe runs (`probe_20261122T045718Z.json` with three HTTP 403 attempts and `probe_20261122T045931Z.json` with three HTTP 200 attempts totaling 1,565 tokens) account for 29,182 tokens; the remaining +2,061 tokens and slight cost delta (`$0.000247`) in the 7-day account window reflect separate exploratory/diagnostic API calls outside the saved benchmark JSONL and differences between per-token snapshot estimates and server-side billing/rounding. The two series are therefore reported separately rather than claimed as a 1:1 itemized ledger.

## LLM benchmark summary (token-priced estimates from `outputs/run_20261122T050023Z.jsonl`)

GMI-hosted models only, not the whole market. All 3 models ran the same 12 text transcript fixtures (`test_cases.json`) at `temperature: 0.0`, `max_tokens: 512`, `chunk_words: 64`, `stream: false`, `timeout_s: 15.0` (`reasoning_effort: "low"` passed for `google/gemini-3.8-flash`).

| Model | Fixture checks passed | Failed cases | HTTP attempts | Latency median / p95 / mean (s) | Prompt / Completion / Total tokens | Estimated spend (USD), after discount / list |
|---|---|---|---|---|---|---|
| `google/gemini-3.8-flash` | 12 / 12 | none | 13 | 2.798 / 18.949 / 4.861 | 3,972 / 549 / 4,521 | $0.005038 / $0.005038 |
| `zai-org/GLM-5.3-Flash` | 11 / 12 | `c12` (timeout, 15 s configured; elapsed 21.998 s) | 13 | 10.090 / 21.998 / 11.852 | 3,703 / 1,651 / 5,354 | $0.000690 / $0.001381 |
| `deepseek-ai/DeepSeek-V4-Flash` | 9 / 12 | `c02`, `c11`, `c12` (cut off at 512-token cap) | 20 | 1.8695 / 6.270 / 2.652 | 7,639 / 7,288 / 14,927 | $0.002022 / $0.003110 |
| **Sweep subtotal (36 cases)** | **32 / 36** | **4** | **46** | **3.803 / 18.949 / 5.876** | **15,314 / 9,488 / 24,802** | **$0.007750 / $0.009529** |
| Gemini repeats (3 cases x 3 runs: `c03`, `c05`, `c11`) | 7 / 9 checks; 2 / 3 cases identical (`c05`, `c11`) | `c03` varied (1 hyphenated miss, 1 timeout) | 9 | 4.807 / 18.732 / 6.953 | 2,399 / 416 / 2,815 | $0.003359 / $0.003359 |
| **Full logged run (45 rows)** | **39 / 45** | **6** | **55** | **4.089 / 18.949 / 6.052** | **17,713 / 9,904 / 27,617** | **$0.011109 / $0.012888** |

Offline fail-closed safety suite (`benchmark_results.json`): 5 of 5 simulated failure modes (`http_500`, `timeout`, `empty_content`, `reasoning_only`, `unrecoverable_truncation`) raised `PolisherError` with zero raw ASR leakage (`$0.00`).

## Media generations (60-second hybrid video + soundtrack)

Output video: `video/voice-polisher-60s.mp4` (`1920x1080`, `30 fps`, `60.000s`, H.264 + AAC 48kHz stereo, `faststart`). Minimal warm-paper (`#F5F3EE`) editorial motion graphics combined with 4 bespoke 1080p Google Veo 3.1 architectural cinema shots and a melodic progressive house instrumental (`minimax-music-3.0`). Zero on-screen sponsorship text.

Post 2 minimal card: `polisher-card.png` (`1920x1080` PNG rendered locally via Chrome from `benchmark_results.json` and `outputs/run_20261122T050023Z.jsonl`, `$0.00`, zero on-screen sponsorship text). Note: the `1920x1080` resolution (16:9) matches the Day 2 benchmark card format rather than the 1600x900 2-number template.

| Asset | Model | GMI `request_id` | Params | QA Status | Cost (USD) | Notes |
|---|---|---|---|---|---|---|
| `video/raw/clip1_capture.mp4` | `veo-3.1-generate-001` | `6bf62114-33df-4854-ac7e-9c3d77ff9e7b` | 1080p, 8s, 16:9, `generateAudio: false` | Approved (1st run) | $1.60 | Brushed-aluminum desktop mic + concentric acoustic ripples in ceramic water dish on warm alabaster |
| `video/raw/clip2_prism.mp4` | `veo-3.1-generate-001` | `270c2bdc-f918-436f-8602-84562d74463e` | 1080p, 8s, 16:9, `generateAudio: false` | Approved (1st run) | $1.60 | Overhead optical glass prism aligning scattered charcoal tiles into an ordered row with one terracotta tile |
| `video/raw/clip3_zero_dial.mp4` | `veo-3.1-generate-001` | `3831ea0b-48da-498d-9b80-c5bd12b72f83` | 1080p, 8s, 16:9, `generateAudio: false` | Approved (1st run) | $1.60 | Precision knurled aluminum & matte cream horological dial locking onto single terracotta zero notch |
| `video/raw/clip4_release.mp4` | `veo-3.1-generate-001` | `f76eff46-1122-43bc-9ac8-89138fedcedd` | 1080p, 8s, 16:9, `generateAudio: false` | Approved (1st run) | $1.60 | Minimal brushed-aluminum mechanical keycap depressing & releasing into sweeping golden sunlight |
| `video/raw/progressive_house.mp3` | `minimax-music-3.0` | `0a449059-28bd-4fd1-b326-028cfbe115fd` | 44.1kHz, 256kbps stereo MP3 | Approved (1st run) | $0.15 | Prompted as 124 BPM melodic progressive house instrumental; cut timestamps chosen from measured 4s RMS energy windows in 0-60s |
| **Media subtotal** | | **5 calls** | | **5 of 5 approved, 0 re-rolls** | **$6.55** | `$3.35` under the `$9.90` approved max cap |

## Notes

- Scope: GMI-hosted models only, not the whole market. This log does not compare GMI with any other provider.
- Cash balance (`balance_usd`) and promotion credit (`promotion_credit_usd`) are separate pools and are never summed.
- Benchmark results reflect one run of 12 text transcript fixtures on 2026-11-22 under a 512-token cap and check substring/ending rules, not general model quality.
