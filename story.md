# trAIde — the story so far

Running journal of the build: progress, challenges, and the moments worth telling. Raw material for
blog posts and social posts. Keep entries short and dated; put numbers in, they age well.

- Dashboard (live): https://sandboxes.live/traide
- Code: https://github.com/abozaralizadeh/trAIde
- Posts: `medium_post.md` (Sep 2026 — "I Gave an AI $70 and My Exchange Keys")

## Screenshots to capture (for posts)

Save under `docs/story/` and link here. Dashboard discloses % and R only — never balances.

- [ ] Equity curve, ALL view (the honest post-repair curve)
- [ ] Strategy edge panel — per-family verdicts + risk bars
- [ ] Exit discipline panel — model's closes vs trailing-stop exits
- [ ] Macro calendar panel during a blackout (next: CPI / FOMC day)
- [ ] A "recently closed" card where a tiny win precedes a big move (AAVE / NEAR, Sep 2)
- [ ] Telegram supervisor chat
- [x] Dual run panel — preview with SAMPLE data, pre-launch: `docs/story/2026-09-30-dual-run-panel-preview.png`
- [ ] Dual run panel with real data after a week of `JEV_MODE=shadow` (LLM vs Jev, same window)

## Timeline

| When | What happened | Number worth quoting |
|---|---|---|
| Dec 2025 | First commit. LLM trading agent on KuCoin, real money, tiny account. | ~$70 |
| Jul 2026 | Why it bled: stops inside the noise band, cost model inflated 12x, trail arming on noise. | — |
| Aug 2026 | Built the signal-edge probe. First reading was fake. | +1.24%/15m @ 92% hit -> really −0.007% |
| Aug 2026 | Overlapping samples inflated significance. | t = 3.66 -> 0.51 |
| Aug 2026 | Kelly stand-aside: zero edge -> zero stake, per strategy family. | continuation: 10% win, −7.4R |
| Sep 1 | Give-back leak found. | +31R peak excursion, −4.5R realised; 2.25R target hit 0/81 |
| Sep 2 | Model re-deciding trades every few minutes. | 0.5% wake trigger vs ~2.2% stops; 13-min median hold |
| Sep 3 | 16h outage from one type annotation (`List[Dict[str, Any]]`). | 563 tests green, bot dead |
| Sep 4–14 | Five "freezes": up but not trading. | 31 runs / 0 orders; a gate refused the only 2 valid setups 11x each |
| Sep 8 | Tweet reply found a hole my fix missed: freshness field refreshed itself. | 63h-old data, reported age 0s |
| Sep 9–10 | Equity curve lies, twice. | +72,546,760% displayed; later −0.27% shown vs −16.5% real |
| Sep 15 | First real win — for the wrong reason. | CAP +1.81R "funding carry", zero funding collected |
| Sep 17–19 | First good stretch, in a broad rally. | 28 closes, 79% win, +9.6R; ~$67 -> $70+ |
| Sep 19 | Rally reconstruction: trend playbook benched on the wrong clock. | 143 blocked calls: +1.27% net @240m, 76% hit |
| Sep 19 | New rule: decisions must follow the market. Reverted a rally-tuned change. | helped 1 of 20 trades |
| Sep 19 | The AI assistant's own analysis bug: futures vs spot candle column order. | "early exits cost 2.6R" -> actually helped +0.84R |
| Sep 19 | Exit scoreboard blamed the model for the trailing stop. | 16 trail exits vs 1 model close |
| Sep 20 | Test suite was starting a LIVE trading loop on every run. Found by counting threads. | — |
| Sep 20 | Dependency refresh, LangSmith deprecated endpoint fixed, deploy script installs deps. | 734 tests |
| Sep 19–22 | Second good stretch, rally continues. Continuation trading on measured edge (t≈2 at its own 240m horizon). | 40 closes, 82% win, +11.3R; equity ~$70 → $74.6 |
| Sep 22 | Model's own close on a long→short flip was invisible to its scoreboard (stamped with the new short's lifecycle). | 1 case, fixed |
| Sep 22 | Trailing stop cut 30 winners early in the rally — but every exit is trend-tagged; no chop data yet, so no adaptive trail shipped. | −10.77R vs brackets, all `trending` |
| Sep 22 | Restart audit: only one thing lived solely in-process — the protection R-anchor. A winner already at breakeven before a restart lost its trail for good. Now re-seeded from the recorded trade. | 1 silent failure mode, closed |
| Sep 23 | "The better the bot traded, the blinder its macro guard became." Winning = always holding positions = research never ran = calendar never refreshed. Code now schedules it. | 67h stale, 1 event; CPI + FOMC missing |
| Sep 22–23 | The fix for one outage caused the next. The retry transport added for the Sep 14 dead backend pinned ONE connection pool for the whole process, but every agent run is its own event loop — run #2 tried to close run #1's socket on a dead loop. First run after deploy worked; every run after died. Protection kept guarding open positions the whole time. | 41 failed runs in a row, ~28h without the model |
| Sep 23 | The dashboard said "crash" during a winning week. 1D is only two daily points, autoscale stretched a −0.35% day across the whole chart, and the line was coloured vs the June start — so a +12.4% week drew red. Now coloured by the viewed range, with a return badge and a minimum scale. | 1W +12.41%, 1D −0.35% |
| Sep 25 | The "paying" playbook was a price-feed mismatch. Probes started on the futures price but were scored on the spot price; on ONE-USDT the two were ~30% apart, so the gap counted as profit for funding carry. On futures it is a coin flip. No P&L gain claimed (the trades it would have skipped actually won +1.74R); the fix is about honest numbers. | +25% "5-min return" on a +2.5% move; t=1.90 → ~1.0 |
| Sep 25 | The carry hold waited for the wrong bell. It held funding trades until the next 8-hour settlement, but most contracts pay every 4 hours and the extreme ones (the whole reason for the trade) every hour. So the trail was switched off for up to 7 hours after the payment had already landed. It now reads each contract's own clock from the exchange. | 15 of 18 carry trades on 1h/4h clocks; ONE-USDT's hourly rate read ~8x too small as "%/8h" |
| Sep 25 | A restart-safety fix from three days earlier had never run once, and fixing it the obvious way would have been worse. The saved peak was in dollars divided by contracts, which is 10x to 520,000x too big on some coins, and its key never matched, which hid that. Now it is stored as a price, reset on add-ons and partial closes, and applied once. A restart now gives the exact same decisions as no restart. | 0R realised impact; ~35% of closes were on the coins it would have wrongly closed |
| Sep 25 | The model kept closing trades on the exact condition it had opened them in. The prompt said "if 15m and 1h both turn against you, it's a reversal" but never said *compared with when*, and the model couldn't see its own entry. Carry longs bought on purpose into a 15m/1h downtrend were then closed a few minutes later for that same downtrend. Each open position now shows the model its own entry record, and "fresh" means changed since entry. No push to hold: the only other cases of this pattern were closes that *helped*. | 27 of 31 rule fires were already true at entry |
| Sep 25 | The exit scoreboard graded the model against a system that doesn't exist. It compared each early close with the bare stop/target, but the real bot would have trailed the stop and usually exited much earlier. Replaying the real exit logic on 1-minute futures bars makes the same 5 closes look about half as costly, and the error flips sign in choppy markets, so the scoreboard needed the right comparison, not a fudge factor. | −6.23R vs bracket → −3.44R vs the real stack (n=5) |
| Sep 25 | The bot told the model "this playbook has NO EDGE, it's a coin flip minus fees" nine times in one morning, while the playbook's own numbers said it was *positive* — just not proven yet. The model believed it and repeated it. The rule was fine; the message was wrong, and the scoreboard the model reads never showed the bench at all. Now it says which of the two it is, shows the zero stake before the model proposes, and judges longs and shorts separately: the "edge" was all longs, and three shorts had been sized on the longs' record. | 9 false "NO EDGE" refusals; shorts n=9 at −0.18% sized 0.52–0.79 on longs' n=40 at +1.24% |
| Sep 25 | Every new trade now writes down exactly how it was sized (equity at the time, every factor, contracts after each cap), and every call records which AI model made it and how confident it said it was. The confidence number had seven hard thresholds hanging off it and nobody could check whether it means anything; now it is measured per model, within each day so a rally can't fake it. | 7 confidence thresholds, 0 measurements → measured per model |
| Sep 25 | The prompt was quoting a July experiment as fact: "crossing to fill made +13R, waiting longer lost money, and the fee check picks the good crosses". Re-run on fresh data with the candle columns read correctly, only the first part held, and only for longs in a rally. It also told the model to rest limits 0.5–1.5 ATR away, where the bot's own limits filled 8–33% of the time. The numbers are gone; the model now sees a live table of its own limits by distance (how often they filled, what the fills made) plus a what-if for every call at every depth, checked against reality on 82 of 82 limits. | fill rate 62% under 0.25 ATR vs 8% at 1–2 ATR; 2 of 3 July claims not reproduced |
| Sep 25 | Plumbing that made the bot look busier than it was. Every "are my stops gone?" check after a close failed because the tool sent `DASH-USDT` where the exchange only accepts `DASHUSDTM`; failed cancels were printed as "live order"; and the market scanner kept offering coins this bot can't trade at all (no spot pair — FHE, TAKE, stock perps), which took a third of the short-side slots and 43 wasted tool calls. One shared "can we trade this?" check now drives both the scanner and add-coin, and each scan row says if the coin is quarantined or failed its last data check, with a retry time taken from the data itself (a candle gap has to age out of the window). | 9 failed stop checks, 10 mislabelled cancels, 6 of 17 short-side rows untradeable in one snapshot |
| Sep 25 | The bot could not tell a rally from chop. Its per-coin regime tag said "trending/strong" on 164 of 188 trades and BTC "bullish" on 186 of 188, so the question "is the trailing stop right in chop?" could never be answered. Every entry and probe now records the market state (share of liquid perps up over 24h, their median move, BTC 24h/72h and daily trend strength). It is recording only: the one tempting finding ("high breadth = profit") was one rally week in disguise, so the model sees today's numbers, not a verdict. | chop row for the trail: 0 → accumulating |
| Sep 25 | A review of the day's own changes found the new scoreboard telling the model the wrong thing. It showed one "stake" per playbook, but the bot judges longs and shorts separately, so it said funding carry was fully staked while shorts were refused, and told the model to skip continuation while longs would have gone through at full size. It also told the model not to bother proposing benched playbooks — but a refused proposal is the only thing that can ever un-bench one. Plus: a report-only log line was quietly deciding the live BTC gate, and a hung funding endpoint could hold the order lock for 15s per carry position per poll. | 25 verified findings fixed; stake now per side; 0 network calls under the order lock |
| Sep 25 | The bot's own safety gates had never been graded. When a gate like "don't chase an exhausted daily trend" refused a trade, it left no record at all, and one of them didn't even write a log line. The study that said these gates save money only held because of one rally week. Now every refusal is labelled with the gate that made it and scored, and every market analysis records which gates *would* block each side, even when the model never proposes the trade. That second record catches the model quietly steering around the rules (most of its "no" answers cited the exhaustion gate). The result goes to the dashboard and the owner's Supervisor, never to the trading AI, so it can't learn to argue its way past a gate. | 8 refusals, 0 records → every refusal plus ~6x more "would block" readings; verdict only after 20+ same-day comparisons |
| Sep 26 | The bot benched its own winning playbook in the middle of an alt rally — by rounding. It scores each playbook at the horizon it holds trades for (60m or 240m, picked by the median hold). Continuation's median sat at 118-141 minutes, right on the 120-minute midpoint, and one fast FET winner tipped it to 60m, where the calls look flat. Zero stake for 21 hours while 93 of the top 100 coins rose; the skipped calls were +1.44% four hours later. And because the model now saw the zero stake *before* proposing, it stopped proposing — so no new evidence could ever un-bench it. Now scored over the real mix of hold times, and the model is told to submit benched setups anyway. | 21h flat, 0 trades; skipped calls +0.05% @60m vs +1.44% @240m |
| Sep 26 (pm) | The fix worked on its first run: shown a benched playbook, the model SUBMITTED an honest SOL continuation long instead of declining, and code recorded the call and refused it with the truth ('positive, not yet proven, t=0.76'). It then did it again next run — which exposed the next trap: continuation's evidence was already at its 150-row cap (only 88 independent), so every repeat would push out an older real observation until the verdict forgot itself. Repeats inside the scoring window are now not stored. And the one-off data fix that two deploys skipped now runs inside the deploy script itself. | 3 runs, 2 honest submits, 0 relabels |
| Sep 28 | "It won a lot last week and loses now — why isn't it shorting the drop?" A 12-agent investigation, with every number re-derived by a skeptic, found the question mostly wrong and the answer interesting. The "loss" was −$0.28 over three days, after a +$8.35 week. The alts weren't falling for most of the week (only BTC was); shorting Sep 24–27 would have lost every day. The real change was the new AI model (switched Sep 23): its trend-following longs did worse than just buying the same coins, and once the rally evidence aged out of the store the bot rightly benched them — the benched calls went on to lose. When the real dump came (Sep 28, ~6h of data), shorts hit a lagging daily label that the prompt called "BLOCKED, not optional" even though the gate never refused a single trade, confidence bars the new model's honest ~0.69 never reaches, zero-stake short playbooks, and the model's own refusal to chase an oversold coin. Catching the whole flush was worth ~$1–2. Fix: tell the model the truth about its gates, measure more — don't loosen survival on one morning. Report: `docs/analysis/2026-09-28-why-few-trades-no-shorts.md`. | −$0.28 "loss"; 0 daily-gate refusals; 20 short vs 53 long calls; confidence p50 0.69, 1 of 97 calls at 0.80 |
| Sep 28 (pm) | Fixed the parts of the story that were the bot telling itself something false — without loosening a single safety rule. The prompt's daily-gate text is now generated from the same settings the code enforces (which side it refuses, which doors are open, at what confidence), instead of "BLOCKED, not optional". The "buy the oversold dip" hint now says when another gate will refuse it. Long-only wording got its short-side mirror. Every call now records which code build and which exact prompt made it, so the next model switch can be told apart from the next prompt edit. And the evidence store keeps longs and shorts in separate drawers: a busy long side had been throwing away the short side's history (141 longs, 9 shorts). Found on the way: every placed trade had been counted twice in the evidence (492 rows → 401), and a clean install couldn't even import the bot (a dependency that a library upgrade silently stopped providing). | 0 rules loosened; 91 double-counted calls; 1,213 tests |
| Sep 30 | A second trader joins the same account: **Jev** (typesafe.ai), a classifier that answers in well under a second and returns calibrated probabilities instead of prose. It gets the same market analysis as the GPT agent, but only the opportunity questions (long / short / stand aside, which playbook, enter now or on a pullback, near or far target), and it never sees the account, the gates or its own score. Code builds its stop and target from the chart levels, and every call goes through the exact same order path, so it can't skip a safety rule. Each trader is scored only on its own calls; Jev starts at trial size and has to earn more stake the same way the GPT playbooks do. Shadow mode first: every call scored, no orders. | cost ~$0.04 per million input tokens; 1 open position cap; 0 new safety rules |
| Sep 30 (pm) | Jev went live in shadow — and the first feedback was "I can't tell what it's doing". Numbers without the story behind them. Now every Jev answer can be followed end to end: what it saw, what it said and how sure, what the GPT agent said about the same coin at the same moment, the plan code built, what the safety gates did with it, and how price actually moved 15 minutes, 1 hour and 4 hours later — in the log, on the dashboard and in LangSmith (where the market's answer is attached to each call as a score). | 1 line per call; 3 horizons; 0 new calls from the dashboard to the bot |
| Sep 28 (night) | "Where is the leverage?" The public dashboard never showed it. Now every position and order carries its leverage, and a new tile shows the whole book's real exposure as a multiple of the account, next to the number that actually decides how hard the bot bets: risk per trade. The lesson for the post: on this bot leverage is only how much margin gets posted; the bet size comes from the stop distance and a fixed slice of the account, so turning up leverage changes nothing until you turn up the risk. | risk 0.75%/trade; 3x leverage cap |

## Where it stands (Sep 28, 2026)

Equity ~$75.5 (from ~$67 on Sep 17). Sep 17–23: 87 closes, +24.3R, +$8.35, 79% win — almost all trend longs in an
alt rally. Since then nearly flat (Sep 24–28: +$0.23, −1.1R) and trading 1–4 positions a day instead of ~12: the
trend playbook is benched on the new model's own record (t 0.5–0.8), and every other side trades at explore size.
One regime of wins, one model switch, one six-hour dump — still encouraging, still not proven.

(Sep 20 snapshot: 190 closed trades, +1.2R, about +$1.9 all-time; last week +11.3R at 82%.)

## Recurring themes (good angles for posts)

- **The trading is the easy part; knowing the truth is the hard part.** Most big bugs were
  measurement bugs, not strategy bugs.
- **Code enforces survival, the model owns opportunity.** Every time a hard gate overrode the model's
  opportunity call, it became a silent veto and the bot froze.
- **Every safety gate needs an escape hatch and a counter.**
- **Losing evidence must not look like never having had it** (probe eviction un-benched a bad playbook).
- **Build in public works** — a stranger's one-line reply beat my own fix.
- **The AI that debugs the AI is also wrong sometimes.** Replays must reproduce reality; mutation-test
  every fix; read the raw source, not the sanitized one.
- **Right for the wrong reason** — the "funding carry" edge is really a crowded-positioning signal.
- **Decisions must follow the market** — never ship a constant tuned on the current regime.

## Open questions / next chapters

- Trailing stop that adapts to trend vs chop by itself (evidence: +9R left on the table in a rally,
  ~35R saved by the tight trail in chop).
- Does the funding/positioning playbook survive 50+ samples?
- Do the directional gates earn their keep? The gate scoreboard needs months of same-day comparisons across regimes before any gate should move.
- Equity index has no concept of deposits/withdrawals yet.
- Can a sub-second classifier (Jev) call direction as well as a reasoning LLM on the same data, for a fraction
  of the cost? Do the two agree, and when they disagree, who is right? (dual-run panel, `JEV_MODE=shadow` → `live`)
- Is the new model (gpt-6-luna, Sep 23) a worse trend-follower than gpt-5.6-luna, or was it the Sep 25 confidence
  prompt? Nothing stamps which build made a call yet, so the two can't be separated (Sep 28 analysis, M0).
- Can the bot short a fresh dump without chasing? The daily label lags by days by design; the only escape hatches key on
  a confidence number that no longer ranks calls. Pre-registered in `docs/analysis/2026-09-28-recommendations.md` (S1–S5).
- ~~The +13R "marketable entry" replay needs re-verifying with correct candle columns.~~ Done Sep 25: direction held (rally longs only), the RR-filter and wait-longer claims did not. Next: does the live execution map's `crossBand` ever show sub-floor crosses paying outside a rally?
