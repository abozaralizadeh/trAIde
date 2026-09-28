# Recommendations: few trades and almost no shorts, Sep 24–28

Sources: the verified fact base (`docs/analysis/2026-09-28-facts.md`, cited as §n), three designer lenses and two critiques. I spot-checked the load-bearing line references against the repo and they hold. Numbers that come only from a designer's replay, and are not in the verified base, are marked *(replay)*.

## The short answer

- **The loss is small.** Sep 26–28 lost **−$0.28** (§1). The last 7 days are still **+$2.95**. Every gpt-6-luna close since the Sep 23 switch sums to +$0.25 and −1.05R.
- **Not shorting Sep 24–27 was right.** A plain short on the analysed coins lost at 240m: −0.85%, −0.68% and −0.56% net on Sep 25, 26 and 27. The model's own gate-passing shorts lost −1.45%, −1.77% and −2.05% (§3). Only about 6 hours of Sep 28 (from 02:45 UTC) paid. Catching that whole flush was worth roughly **$1–2** (§4g).
- **"Few trades" is mostly continuation:LONG being benched,** at t 0.51–0.77 (§2). The Sep 17–23 rally evidence was pushed out by the 150-row per-family cap, so the verdict now rests only on gpt-6's own calls. Those calls trailed a plain long on the same coins (§2). The benched calls then went on to lose *(replay: −0.41% net, n=29; the subset checked against the log: −0.20%, n=11)*. The model switch and the Sep 25 confidence prompt (6382eef) are confounds that nobody can currently separate, because nothing stamps which build made a call.
- **Several real defects are worth fixing in any regime:**
  - The prompt describes the daily gate wrongly: "BLOCKED … NOT optional", yet the gate made 0 hard refusals (§4a).
  - Some wording is written only from the long side (§4e).
  - fadeSetup advertises trades that the tf_conflict gate then refuses (§4e).
  - Several measurement gaps mean nobody can say *why* the bot stood aside without a 12-agent investigation (§4c).
- **What we are not doing:** no survival threshold moves on the strength of one six-hour flush.

**Order of work.** M0 (build stamp) and M4(a) go first; both are S effort. M1, M2 and M3 can run in parallel; M2 is time-sensitive. Then the prompt edits, one at a time, at least `_GATE_MIN_DAYS` (3) UTC days apart, so each one shows up as its own build in call mix and confidence reachability: W1 → W2 → W3 → W4. W5 (docs) can ship at any point. M5 lands after M3 and M4. Section 3 is decided only on M-data. Every item needs the full test suite passing and a bot restart to deploy. Add a dated story.md line per shipped item.

---

## Status (updated Sep 28, evening)

Shipped on branch `claude/jolly-curie-9nqogp`:
- **W1** — the daily-gate prompt text is rendered from `cfg.regime` (`agent._daily_gate_rules`); the retry text and the analytics `DAILY GATE` / exhausted-bullish hints now say what code refuses and which routes pass.
- **W2** — `regime.tf_conflict_opposes` is the single predicate (order path, `directional_gates_against`, `fadeSetup.tfConflictRefuses`); "ready-made" wording removed.
- **W3** — mirrored wording (trend in either direction, RSI 65/35, strong band per side, entry headers, the ONDO case mirror with the intact-trend condition kept, research scan text). Pinned by `tests/test_prompt_honesty.py`.
- **W5** — `is_hostile_regime` docstring (side-blind, S2 pointer) and the 1D window comment (~50 completed bars).
- **M0** — `src/buildinfo.py`: `build = {code, prompt}` on signal probes, gate refusals and entries; `BUILD:` line at startup.
- **M1 (part)** — retention per (family, side); `signal_probes()` twin dedupe (a placed call no longer counts twice: 492 → 401 rows on the Sep 28 store, no stake verdict changed). **Not done:** folding orphan order rows into the bucket and dropping the trades union (orphans still come from `trades` and still vanish when `MAX_TRADES` prunes them).
- Also: `httpx` added to requirements.txt (openai ≥ 3.19 no longer installs it; a clean venv could not import `src.agent`).

Not started: W4, W6, W7, W8, M2–M6, and everything in section 3 (by design — evidence first).

## 1. Do now: honesty and wisdom fixes (no gate or stake changes)

### W1. Describe the daily gate exactly as the code enforces it (PR1, modified). P1, ships first and alone after M0
- **What.**
  - Rewrite the STEP 2 daily-gate block to cover three things: what the gate reads (completed UTC bars only), what it gates (plain entries against a non-neutral `daily_bias`; exhausted or weak dailies gate neither side), and which routes it admits.
  - Build the route list from **cfg**: thresholds, `reversal_*_enabled`, `reversal_*_require_15m`, `declared_setups_enabled`/`declarable_setup_families`, `fade_extreme_enabled`, `trend_shorts_enabled`. Never advertise a disabled route. Point to the setup_family section's existing anti-relabel text rather than restating a list of doors.
  - Fix the retry text.
  - Add `intraday_bias`/`intraday_strength` summary keys.
  - The DAILY GATE hint must say that tf_conflict is then true by construction, so any side the 15m opposes is refused.
  - **Keep the true judgement**, labelled as judgement rather than code:
    - Exhausted-bullish daily: "a short here is counter-trend; only on a real reversal or extreme".
    - Exhausted-bearish daily: "do not chase a fresh dump; the trend-short is for a rally/retest".
- **Why.**
  - daily_opposing made **0 hard refusals**; its whole effect was the model not proposing (§4a).
  - The prompt contradicts itself: agent.py:1680-1681 says "BLOCKED … NOT optional", while agent.py:1793-1803 says the setups pass "on your call alone".
  - 8 of 20 short calls passed a bullish daily through hatches (§4a).
  - The model cited a "bullish daily" on exhausted coins where no daily gate exists (verifier).
  - 0.80 is hard-coded at :1683/1685, so it can drift from `REVERSAL_*_MIN_CONFIDENCE`.
- **Where.**
  - `src/agent.py:1679-1688` and `:2012`.
  - `src/analytics.py:903-908`: add keys only. `overall_bias`/`strength` must stay unchanged because `_relative_strength_exception` and `_alt_long_is_blocked` read them.
  - `src/analytics.py:985-999`: keep the tokens pinned at `tests/test_analytics.py:148,176,196`, and no "Wait" (:237).
  - The tf_conflict construction is at `analytics.py:920-923`.
- **Effort** M.
- **Risk.**
  - More counter-daily shorts in dips that reverse. Shorts on exhausted coins lost −0.77% @240m, n=151 *(replay)*.
  - Some temptation to relabel trades as breakout.
  - Bounded: every gate, floor and stake is unchanged.
- **Watch.**
  - "Blocked by the daily" narratives on coins where daily_bias is neutral should go to ~0.
  - Declared-route calls per side, via gatesPassed.
  - Breakout calls whose rationale names no level break.

### W2. Make fadeSetup honest about tf_conflict, using one predicate (PR5, modified). P1
- **What.**
  - New `regime.tf_conflict_opposes(conflict, bias_15m, side)` used by the order path, by `directional_gates_against` and by fadeSetup (`tfConflictRefuses`). This removes a copy rather than adding one.
  - New note: when declared, a fade is admitted past the daily and 1h gates. tf_conflict has no fade route, so when `tfConflictRefuses` is true the code refuses it, and that is a gate refusal, not fade evidence.
  - In agent.py:2467-2469, drop only the "ready-made" framing: "entryMap.fadeSetup flags a 15m RSI extreme; check fade_extreme's side stake and tfConflictRefuses."
  - Make the README:408 invariant true by extending its test to tf_conflict.
- **Why.** TAO×2 and FIL fade longs were pushed toward "FADE-EXTREME ALLOWED" and then got "TF CONFLICT BLOCK" (§4e, verifier E12). The prompt calls fadeSetup ready-made while fade was staked at 0 on both sides.
- **Where.**
  - `src/tools.py:4243-4268` and `:~532`.
  - fadeSetup is assembled at `src/tools.py:~2776`. Compute the flag after `fade_setup_available`; do not change its signature.
  - `src/regime.py:997-1003`: keep the `setup_family='fade_extreme'` token (`test_regime.py:984`).
  - `src/agent.py:2467-2469`, README.md:408.
- **Effort** S-M.
- **Risk.** Low, because the gate is unchanged. The refused fade longs lost −1.08% @240m, so this saves turns, not dollars.

### W5. Docs honesty: the hostile throttle is side-blind, and the daily window comment is wrong (SC1 + SC6, merged with PR2(c)). P1, no behaviour change, ship anytime
- **What.**
  - The `is_hostile_regime` docstring should say it is a deliberate **regime** brake. It is side-blind, so trend-aligned shorts on a bearish daily also need 0.75 and get the 0.6 size factor. Name the side-aware form and point to its switch condition (S2).
  - Put the class numbers in a code comment only: n=20, −3.88R *(replay)*, and **mostly Jul–Aug, before the Sep 1 give-back and noise-band stop fixes**.
  - Align README and .env.example.
  - Replace `# 30 days` with the correct 50-bar statement, citing the EMA26 and Wilder ADX seed weight (~2% at 49 bars vs ~11% at 29), so nobody "fixes" it by shortening the window.
  - Note at the daily gate that the completed-bar label cannot react intraday by design, and that the reversal hatch is the designed escape.
- **Why.** §4b: side-blind floor, and all 13 confidence_floor refusals were shorts. §4a: the comment says 30 days but 50 bars are fetched.
- **Where.** `src/regime.py:27-29`, README.md:392, `.env.example:303-304`, `src/tools.py:2614`, `src/analytics.py:~887-904`. Optional test: the 1D request is ≥50 bars.
- **Effort** S.
- **Risk** none.

### W3. Symmetric long/short wording (PR4 + PR3(c)(d), modified). P2, one prompt change
- **What.**
  - (a) "Trade STRENGTH" becomes "Trade the clearest trend, whichever direction: relative strength for a long, relative weakness for a short."
  - (b) The rejection-confirmation line uses the **65/35** pair already used in the same prompt (agent.py:1810/1816): short RSI > 65, long RSI < 35. This adds no new number.
  - (c) Strong-setup band: 40–65 for a long, 35–60 for a short.
  - (d) The research scan text states that 'momentum' covers absolute movers in both directions and drops "in a bearish BTC daily". **No extra per-run scans.**
  - (e) SHORT/LONG entry headers become "at or above / at or below current price".
  - (f) Mirror the ONDO bullet for a breakdown that won't bounce, and **keep "gated to an intact trend (daily+intraday aligned)" in both directions**.
  - (g) Add a source-grep test for mirror pairs.
- **Why.** §4e: long-framed texts, and the 65/40 asymmetry. The BTC-conditioned short scan never activates because BTC's daily never flipped.
- **Where.** `src/agent.py:1580-1584`, `:1808/1814`, `:1853-1854`, `:1883-1891`, `:1894`, `:2130-2131`.
- **Effort** S.
- **Risk.** Low. Hygiene only, with no P&L claim; the rally longs were earned under the old text. The 65/35 choice slightly tightens a long "prefer" rather than loosening a short one.

### W4. Label judgement vs code, and correct the carry headline (PR7 modified + PR6). P2
- **What.**
  - The TP rule becomes: "TP at a structural objective you honestly expect the move to reach (a backstop; most exits come from the trail), SL at true invalidation; if no honest objective clears the floor from your entry, SKIP." Do **not** add "or the next one beyond it".
  - The breakout volume rule becomes "prompt guidance; code does not verify breakouts; 1.5× is the textbook bar, weaker volume is weaker evidence", not "REQUIRED … decline".
  - The carry headline becomes "the one playbook whose payoff is partly mechanical — only while open at a settlement; its own by_family row says whether it pays". Update the matching comment on the tools.py fundingSetup code ("best-documented edge").
- **Why.**
  - §4d: agent.py:1928-1930 contradicts :2385-2390.
  - The model treated the 1.5× volume rule as a code gate (BTC declined twice).
  - story.md, Sep 25: the carry "edge" was a spot/futures feed mismatch (t 1.90 → ~1.0).
- **Where.** `src/agent.py:1928-1930`, `:1749-1750`, `:2394`, and the tools.py fundingSetup comment.
- **Effort** S.
- **Risk.**
  - The PUMP longs that the strict TP reading declined would have lost.
  - Carry was the only short winner (FOLKS +0.91R), so watch that carry calls don't fall while carry side rows are positive at t≥1.

### W6. `summary.dailyContext`: how stale the daily label is (PR2 modified). P2
- **What.**
  - One helper `_daily_labels()` shared by `summarize_multi_timeframe` and a new pure `daily_label_context(raw_df, closed_df)`. It returns lastClosedAt, hoursSinceClose, todayPct, and `ifClosedNow` (the same rule applied to closed bars plus the forming bar).
  - Detect the forming bar by comparing lengths, not by assuming one exists.
  - The note must say `ifClosedNow` is not what the code enforces.
  - `ifClosedNow` is **never read by any gate or by directional_gates_against**.
  - Recording only: add `daily_if_closed_now` to the `_gate_regime_stamp` whitelist and a sanitised `ifClosedNowDailyBias` parameter on `record_signal_probe`.
- **Why.** §4a: a −5.6% day leaves ~90–99% of coins "bullish", and the model cannot see that the label excludes today.
- **Where.** `src/analytics.py:887-904`, `src/tools.py:2656-2676` (both frames are already in hand, so no new I/O), `src/memory.py:634`, `:1870`, README.
- **Effort** M.
- **Risk.**
  - It mostly tells the model the daily has *not* turned, which is accurate.
  - The model could misread it as the enforced label. Falsifier: rationales saying "daily has flipped" when daily_bias had not.

### W7. `entryMap.leg`: how deep this 15m leg has actually retraced (PR3 (a)(b)(f)). P2
- **What.**
  - Pure `regime.leg_structure` over the analysed closed 15m bars, in the coin's own ATR, with no thresholds: fromExtremeAtr, maxCounterMoveAtr and counterMoveHigh/Low for each side, mirror-exact.
  - An entryMap note says a rest beyond the leg's deepest counter-move is a bet that the leg ends.
  - Add the counterMoveHigh/Low bullet to the resistance/support lists.
  - Stamp `legCounterMoveAtr` in `probe_execution_stamp` through a new whitelisted key.
  - Report-only split of the execution map by resting distance ÷ legCounterMoveAtr.
- **Why.** §4d: dump short limits rested at a median 1.30 ATR and 0/5 were reached. The model had no measure of retrace depth. This applies to both sides (median rest: long 0.95 ATR, short 0.98).
- **Where.** `src/regime.py` (next to `overextension_atr`), `src/tools.py:2667` (keep the 15m frame), `:2765-2785`, `:642`, `src/memory.py:1870`.
- **Effort** M.
- **Risk.** The execution counterfactual is negative at every depth (§4d), so more fills are not obviously better. Falsified if fill rates don't differ across the ≤1 vs >1 split after ≥40 limits per side.

### W8. Show one-lot risk before the call (SC7 modified). P3
- **What.** A pure `one_lot_risk(contract, entry, atr_abs, stop_floor_mult, fees)` extracted from `_risk_capped_contracts`, so the order path and the display use one number. Use the adaptive stop floor from `_edge_state()['stop_atr_floor_mult']`, not a fixed 2.5×. Show it in analyze_market_context from the **cached** spec and equity only: None when uncached, never raises, never published.
- **Why.** §4f: the BTC breakout short was refused because one lot risked $0.95 against a $0.57 budget, on either side.
- **Where.** `src/tools.py:1230`, `:~1382`, and the analysis summary.
- **Effort** M.
- **Risk.** Low. It must stay cache-only (the no-network rule).

---

## 2. Measurement fixes (report-only; never a gate input, never in the trading prompt)

### M0. Build and prompt-hash stamp on every call (from EV1). P0, ships first
- **What.** Capture the git short SHA once at startup (total: None on failure) and a hash of each run's system prompt. Plumb both through `run_trading_agent` → `build_tools`, and store them as `ctx['build']` on signal probes and gate probes.
- **Why.** Nothing can separate the gpt-6 switch on Sep 23 from the 6382eef prompt change on Sep 25 (§1, §2). Every W-item can move stated confidence and the per-side call mix; 6382eef moved it by ~10× *(replay)*.
- **Where.** `src/main.py` (startup), `src/agent.py` (run), `src/memory.py:1870`, `:2066`.
- **Effort** S.
- **Risk** none.

### M1. Retain probes per (family, side), and stop the trades table from moving verdicts (EV2 + SC5 merged). P1
- **What.**
  - Bucket key becomes `(_probe_family(row), positionSide)`, with `MAX_PROBES_PER_FAMILY` per bucket and no new constant.
  - `signal_probes()` reads the bucket only; drop the `trades` union.
  - Add a one-time, idempotent fold of the 6 orphan trade rows. It honours the repeat gap, so the KCS repeat stays out. Test it on copies only, and check it against `setup_service.sh`'s pending-row rule, or every deploy re-stops the bot.
  - Move `dashboard_publisher._family_labels` to `limit_entry_records()`/trades.
  - Update the memory.py:33-43 comment, the CLAUDE.md wording ("per-family cap keeps the NEWEST 150"), README and tests together.
- **Why.**
  - §4c: continuation holds 141 long / 9 short, so a busy long side evicts shorts. That is the Sep 14 principle ("losing the evidence ≠ never having it") one level down, now that stakes are per side.
  - Live, continuation:short went from n=4 to n=2 with no new call, purely from MAX_TRADES pruning (verifier reproduced it).
  - The (symbol, ts) dedupe never matches: 0 of 97 trade rows share a timestamp with their probe, which is 2–7 s earlier. So any consumer that doesn't de-overlap double-counts placed calls.
- **Where.** `src/memory.py:548-572`, `:2907-2935`, `src/dashboard_publisher.py:~306`, `scripts/resettle_probes_futures.py`, `setup_service.sh`, `tests/test_memory.py:1139-1181`.
- **Effort** M.
- **Risk.**
  - A quiet side's old verdict becomes stickier (fade_extreme:short rests on Sep 2–16 data), so its age must be visible (M5).
  - The worst-case file bound doubles.
  - No stake changes on today's store *(replay)*.
  - Property test: n per family:side never decreases between two reads with no new call on that side.

### M2. Gate-state fold: structural sub-cells plus a day-level market aggregate (SC3 modified). P1, time-sensitive
- **What.**
  - In each folded day/side cell, add a separate `sub` map, not new gate keys:
    - `daily_opposing@turn`: blocked, with 1h and 15m both agreeing with the side.
    - `anti_fomo@turn`: mirrored.
    - `hostile@aligned`: a short on a bearish, non-exhausted daily.
    - `hostile@counter`.
    - A small fixed cross-tab: 1h agrees × 15m agrees × exhausted.
  - Score each sub-cell against the **parent gate's allowed rows**. `total − key` would be the wrong baseline.
  - Stamp each day/side cell with its mean breadth24, mean basket-median 24h and n.
  - Keep all of this out of `SCORED_GATES`, `directional_gates_against` (its pin test) and agent.py.
- **Why.**
  - §4a: 12 of 12 daily_opposing-blocked dump shorts had 1h+15m bearish. The scoreboard can only score the whole gate (§5), not the hatch structure S3 would change.
  - `fold_settled_gate_states` drops raw rows and discards marketState, so **every day before deploy can never be re-split**.
- **Where.** `src/edge.py:1615-1639`, `:1713-1753`, `src/memory.py:2196`, `:634`.
- **Effort** M.
- **Risk.** Small payload growth. Sub-cells are thinner than their parent.

### M3. Stamp what the order path actually decided: stakeAtCall plus post-probe disposition (EV1 modified). P1
- **What.**
  - `stakeAtCall = {judgedOn, sideThin, sideN, n, reason, tStat, stake}`, taken from `_stake_fl`/`_family_scale_fl`, the values the order path already has. **No second `family_stake` call**, which would silently disagree when the explore factor ≠ 0.4.
  - New `memory.stamp_probe_outcome(symbol, ts, outcome)`, with `record_signal_probe` returning its ts. It records the stand-aside reason, NET RR, min-lot/sizing/concentration refusals, the placed clientOid and the fill.
  - New `edge.stand_aside_scoreboard`: per family:side, admitted vs refused "no edge" vs refused "unproven" own-row vs refused thin→pooled. Built on `family_stake_status` and `_probe_observations` (never a copy), with mix legs reported per horizon, day-clustered, with the `_GATE_MIN_DAYS` rule and a fill-conditioned split.
  - Test: nothing feeding `family_stake_status` reads `stakeAtCall`.
- **Why.** No probe records its stake or outcome, so the investigators had to rebuild every board by replay, and that failed wherever rows had been evicted (§4c). S1 and S5 can only be decided on this data.
- **Where.** `src/tools.py:4366`, `:4443`, `:4480`, `:~4535-4700`, `src/memory.py:1870-2027`, and `src/edge.py` (new function).
- **Effort** M.
- **Risk.** About 100 B per probe. Repeats are not stored, so they are graded through M5's counter, not per row.

### M4. Confidence evidence: stamp the floor on refusals and report bar reachability (SC2 modified). P1, part (a) ships with M0
- **What.**
  - (a) Add `minConfidence` and `hostileRegime` to the confidence_floor refusal dict, through `_record_gate_refusal` → `record_gate_probe`. Backfill the 13 existing rows on a copy, flagged `minConfidenceReconstructed`.
  - (b) `confidence_edge_stats(refused_probes=…)` reports a `withLowTail` block labelled **"partial low tail (self-censored)"**. It uses one shared `_probe_observations` pass, and the headline verdict is unchanged.
  - (c) Per model and build: the share of calls at or above each **configured** bar, read from cfg.
- **Why.** §2: stated confidence is p50 0.69 and 1 of 97 calls reached 0.80. §4b: every turn-time hatch sits at 0.75–0.80. The pre-change p50 of 0.76 vs 7% at ≥0.75 afterwards is a *(replay)* comparing different populations. Since the prompt says "only place a trade if confidence ≥ floor", the refused rows are a self-selected tail, not an uncensored sample.
- **Where.** `src/tools.py:4312`, `:1161-1188`, `src/memory.py:2066-2110`, `src/edge.py:1473`.
- **Effort** S-M.
- **Risk.** Reachability ("the bar is hit 1% of the time") is the most damaging number to leak to the model. Keep it out of `confidence_edge_for_prompt` and edgeReport (see M5 guards).

### M5. One owner-only "evidence health" block, not five boards (EV6 + SC8 watch + EV1 grades + SC2(c), consolidated). P1, after M3 and M4
- **Contents,** per family:side and built only on the live functions:
  - **Supply funnel (24h):** calls stored (counted from the `signal_probes` bucket, not the union); repeats not stored, from a **separate small counter** so evidence rows are never mutated; pre-probe refusals by gate, including confidence_floor ("calls lost to the floor"); stand-asides by reason; post-probe refusals, placed and filled (M3).
  - **Provenance:** oldest and newest observation, age of the newest, share by model and by build, and share since the last model change.
  - **Precision:** the achievable SE at the retention cap next to each side's net. That is EV4's one real failure mode: at the cap, an edge below ~0.24% can never release.
  - **Cross-key loss:** the count of observations dropped because another family or side held the (symbol, horizon) window (EV3's report).
  - **Grades and cells:** stand-aside grades (M3), reachability and withLowTail (M4), M2's @turn and hostile@aligned cells with their day count and up/down-day coverage, and realized stack-scored R by class.
  - **Decision table:** the section 3 criteria, as text in README next to "Regime throttle".
- **Guards (required):**
  - Extend the Supervisor don't-relay text (`src/supervisor.py:45-47`, `:69-74`, `:184-194`, `:465-472`) to every new key. Permanent notes enter the trading system prompt (`src/agent.py:2089-2094`), and the current agent.py grep test does not cover that path.
  - Add a test that asserts on the actual edgeReport and analyze_market_context payload keys, not only on the agent.py source.
  - Extend the grep at `tests/test_tools.py:1277`.
- **Where.** `src/supervisor.py:34-65`, the dashboard whitelist, and a new hourly line beside `src/main.py:516`.
- **Effort** M.
- **Risk.** Log and payload volume. The plain long/short baseline (from gate_state_days) must stay owner-only.
- **Validation.** On the Sep 28 copy it must reproduce the verified short funnel: 20 calls, 10 floor refusals, 5 stand-asides, 1 min-lot, 4 placed, 1 filled (§4). Any mismatch is a bug.

### M6. Name the one survival gate that sits before the probe (both critics' "missing"). P1, S
- **What.** A comment at `src/tools.py:4309` and a CLAUDE.md bullet. confidence_floor refuses **before** `record_signal_probe` (:4443), so floor-refused calls never feed their side's own row. That is in tension with "a stood-aside side re-opens ONLY through new calls". It is measured in M5 and decided in S4.
- **Why.** §4b: all 13 floor refusals were shorts, and 10 of the 20 short calls were refused there. The self-censored share is unmeasured.
- **Risk** none.

---

## 3. Survival-rule changes that need evidence first

**The evidence standard, defined once and used by every item below (R\*):**
- ≥ 20 distinct UTC days in the row or cell.
- The days are split **definitionally** by the sign of that day's mean basket-median 24h (from M2's fold aggregate). Each half has ≥ `_GATE_MIN_DAYS` (3) days, and each half's differential has the same sign as the pooled verdict.
- The pooled verdict clears tCritical at df = days − 1, the scoreboard's own test.
- **Not** "covers ≥2 breadth24 terciles". Those cut-points are recomputed from the same rolling window, so almost any 20 days span two terciles by construction, even inside one rally.
- These are rules for a **human** code change and never a runtime input.

| # | Change (source) | Evidence to watch before shipping | Where / effort / risk |
|---|---|---|---|
| S1 | **Thin-side rule B**: a side with <20 own observations explores at 0.4 instead of inheriting the pooled stand-aside, keeping `family_explore_factor`'s thin cap so the Sep 24 bug stays closed (EV5). | M3's "refused thin→pooled" row: ≥20 de-overlapped observations meeting R\*, with fill-conditioned net > 0 and not worse than admitted thin-side calls by >1 SE. Keep the current rule if that net is ≤ 0. Correction: edge.py:405 is n<20 "insufficient data", not "unproven", so today's rule is internally consistent; there is no consistency argument for changing it. | `src/edge.py:329-368, 405-427`. S. Loosens a stake rule. In this window it was $0 (0 of 3 limits reachable). B would re-open range_edge:long (own −0.38%, n=10). |
| S2 | **Side-aware hostile floor**, `hostile(side) = exhausted OR daily opposes side` (SC1's switch). | M2's `hostile@aligned` short cell with blocked ≥ baseline meeting R\*, **and** ≥20 *new* trend-aligned-short closes with stack-scored mean R ≥ 0. | `src/regime.py:27-43`, `tools.py:4309`. S. Loosens trend-aligned shorts. The only realized record is −3.88R over n=20 *(replay, mostly under old exits)*. |
| S3 | **Structural counter-daily hatches**: drop the 0.78/0.80/0.72 confidence tests, keep 1h+15m structure, cap at explore size via a `_structural_hatch` flag (SC4). | M2's `daily_opposing@turn` shows "blocked outperform" **per side** meeting R\*, **and** M4's withLowTail is not "informative". That is necessary but not sufficient, because the tail is self-censored. Before shipping: decide whether the 0.75 hostile floor moves too. Otherwise only reversal SHORTS on non-exhausted bullish dailies (and the deadlock hatch) open. Version the state-instrument key, because `directional_gates_against` changes meaning. Remove the config fields (dataclass **and** loader, README, .env.example) only in the same change. | `src/regime.py:129-171, 1384-1481`, `tools.py:4144-4210`, config. **L.** Revert if realized structural-hatch closes reach mean R < 0 at t ≤ −1 over ≥20 closes, or the hatched differential turns significantly negative. |
| S4 | **Floor-refused calls as side evidence** (both critics' "missing"): tag them `belowFloor` and let them count toward their side's own row, while the floor still refuses the order. | M5 shows a stood-aside or thin family:side whose floor-refused calls alone would reach n≥20 while its stored calls stay <20 over a span meeting R\*, i.e. the floor, not the model, is starving it. This overrides the deliberate "refused calls never move a verdict" design (memory.py gate-probe comment), so it is the **owner's** decision. The floor itself does not move. | `src/memory.py:2907`, `tools.py:4309-4443`. M. |
| S5 | **De-overlap key per (symbol, family, side)** for family rows only (EV3). | M5's cross-key-loss count moves some family:side across n=20, or changes a replayed stake, in both the up-day and down-day halves; and a per-symbol cluster-bootstrap SE is not materially larger than the reported SE. List every stake flip it causes: on today's data, breakout goes 0.4 → 0 on both sides. Ship together with the README "Overlapping probes" paragraph (README.md:406). | `src/edge.py:780-831`. M. It breaks a documented anti-inflation rule, so there must be evidence first. |

---

## 4. Explicitly do NOT do

| Don't | Why |
|---|---|
| Loosen the continuation:long stand-aside or the stateless t<1 bar (`edge.py:425`) (EV4). | That bar is the design. Argue it on principle, not on this episode: "the eviction protected us" (benched calls −0.41% *(replay)*) is one post-rally sample and would cut the other way after the next chop-to-trend turn. Watch the achievable SE at the cap instead (M5). |
| Enlarge or time-window `MAX_PROBES_PER_FAMILY`, "keep the rally", or scope verdicts by model. | Count-not-clock. The Sep 14 un-learning released fade_extreme, which then lost. Model scoping would release fade_extreme:short and range_edge "no edge", and pre-Sep 25 probes carry no model stamp anyway. |
| Let gate probes into signal verdicts (except via the pre-registered S4). | The memory.py gate-probe rationale: a refused call must not move a verdict or evict evidence. |
| Lower `REGIME_CAUTION`/`REVERSAL_*`/`TREND_SHORT`/`DEADLOCK_MIN_CONFIDENCE`, or touch daily_opposing (SC0). | §3: shorts lost Sep 24–27. §4a: 0 hard refusals, so loosening changes nothing directly. §5: nothing significant, one regime. |
| Any breadth/marketState rule in the prompt or the code (PR8). | It would have shorted the Sep 26 and Sep 27 dips, which reversed, for one six-hour flush (§3). marketState stays recording-only. |
| Reword the confidence instruction so the 0.75/0.78/0.80 bars become reachable (PR9). | It re-inflates stated confidence and corrupts the only per-model informativeness measure (within-day rho −0.149, §2). |
| Prompt or refusal text about the continuation:long bench, or asking the model to stop proposing benched sides (PR10). | The Sep 25–26 freeze: a decline records nothing. Repeats are already not stored. |
| Remove "after a rally/retest" from the exhausted-bearish trend-short hatch text (PR3(e)), or drop "intact trend" from the ONDO case. | That hatch is already asymmetric in favour of shorts: exhausted-bullish continuation longs have no hatch (agent.py:1756; §6 anti_fomo). Removing the anti-chase line widens that asymmetry and is fitted to RAVE on Sep 28. |
| Add "a continuation candidate on the other side" to fadeSetup, or sharpen "stake above zero" in the spend-your-turns line (PR5 parts). | It nudges the model to chase the leg at every RSI extreme; the chase would have lost in the reversing dips. The extra emphasis sits right after "SUBMIT benched setups". |
| TP wording "or the next one beyond it" (PR7(a) part). | It invites the TP-pushing that the same prompt forbids. The strictly declined PUMP longs would have lost. |
| Shorten the daily window to 30 bars, or let the forming bar into daily_bias or any gate. | It degrades EMA26/ADX seeding and breaks the no-repaint policy. `ifClosedNow` is display only. |
| Mandate extra "losers"/"short" scans every research run (PR4(d) part). | 'momentum' already sorts by absolute move with `side='both'` (§6). It would add tool calls and no information. |
| Use rolling breadth24 terciles as a regime-diversity test (SC8/SC4/EV5 criteria). | By construction nearly any 20 days pass it. Use R\*. |
| Put stakeAtCall grades, reachability, withLowTail, the plain long/short baseline, @turn cells or the decision table into the trading prompt, edgeReport, analysis output or a Supervisor permanent note. | Report-only rule. Permanent notes enter the trading system prompt. |
| Ship S1–S5 on the Sep 28 flush. | It is one partial day with overlapping NEAR×5 and SEI×3, worth about $1–2. |

---

## How the proposals and critiques were reconciled
- **Merged.**
  - EV2 + SC5 became M1.
  - SC6 + PR2(c) became W5.
  - EV1 + EV6 + SC2(c) + SC8's watch block became M3 + M5.
  - PR4 + PR3(c)(d) became W3.
  - PR6 + PR7 became W4.
- **Modifications applied.**
  - PR1: cfg-rendered routes and flags, true judgement kept, ships alone after M0.
  - PR5: honest note only.
  - PR4(b): 65/35, taken from the feasibility critic because the pair already exists in the prompt.
  - PR7(a): tightened wording.
  - EV1: stamp existing values, add a disposition stamp.
  - EV6: a separate repeat counter, and counts from the bucket.
  - SC2(b): relabelled as a partial, self-censored tail.
  - SC3: `sub` map, parent-allowed baseline, day-level market aggregate.
  - SC4: scope corrected, key versioned.
  - SC7: shared pure helper, cache-only.
  - SC8/EV5: R\* replaces the tercile test and the "≥3 days" test.
- **Dropped:** PR3(e), PR4(d)'s extra scans, PR5's continuation sentence and "stake above zero", and PR7's "next one beyond". No proposal was rejected by both critics.
- **Added from the critiques:**
  - The build-stamp-first sequencing.
  - The post-probe disposition stamp.
  - M6 and S4 (the floor-before-probe tension).
  - The Supervisor relay-path guard and payload-key tests.
  - The signal_probes dedupe bug, fixed by M1.
  - The dashboard `_family_labels` dependency.

**Suggested story.md line (Sep 28):** "'Lost a lot, no shorts' took a 12-agent look: Sep 26–28 was −$0.28; plain shorts lost Sep 25/26/27 (−0.85/−0.68/−0.56% @240m) and only ~6h of Sep 28 paid (~$1–2 for the whole flush). The real story: continuation longs benched at t 0.51–0.77 after the rally evidence aged out of a 150-row cap, and the prompt telling the model the daily gate 'BLOCKS' trades it never refused once (0 hard refusals). | −$0.28; 0 daily-gate refusals; stated confidence p50 0.69, 1 of 97 calls at 0.80"