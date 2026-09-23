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

## Where it stands (Sep 20, 2026)

All-time: 190 closed trades, +1.2R, about +$1.9 (55% win). Honest equity curve still down since June
(the pre-August losses), but the last week is +11.3R at 82% — one regime, so encouraging, not proven.

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
- Equity index has no concept of deposits/withdrawals yet.
- The +13R "marketable entry" replay needs re-verifying with correct candle columns.
