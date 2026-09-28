# Why did the bot trade a lot last week, and why isn't it shorting now? (Sep 28, 2026)

**Question (owner):** a week ago the agent opened lots of positions and won a lot. This week it opens few positions and
loses a lot. The market was positive last week and it went long. Now the market is negative, so why isn't it opening lots
of shorts?

**Data:** `.agent_memory.json` (snapshot Sep 28 08:42 UTC: 200 closes back to Jul 23, 395 signal probes, 72 gate
probes) and `traide.log` (Sep 27 09:33 → Sep 28 08:42 UTC, 186 agent runs), read against the code at `9c0b583`.

**Method:** 6 independent investigators (P&L, market regime, short-side funnel, code asymmetries, prompt and model
behaviour, evidence loop). Each was followed by an adversarial verifier that recomputed every number from the raw
files. Then 3 designers and 2 critics turned the verified facts into recommendations. The corrected fact base is in
[`2026-09-28-facts.md`](2026-09-28-facts.md) and the full plan is in
[`2026-09-28-recommendations.md`](2026-09-28-recommendations.md).

---

## TL;DR

1. **It did not "lose a lot".** Sep 26–28 lost **−$0.28** (−0.37% of equity). Equity was $75.84 on Sep 25 and is $75.52
   now. The last 7 days are still **+$2.95**. What collapsed is **activity**: from ~12 positions a day to 1–4.
2. **"Few trades" is mainly the LONG side being benched, not missing shorts.** The rally playbook (continuation longs)
   has had **zero stake since Sep 25 13:02**. Its measured edge is now t = 0.51–0.77, which is "positive but not proven".
   That caused 37 of the 42 stand-asides in the log. The rally-week evidence behind the old t≈2 was pushed out by the
   150-row per-family store cap. The verdict now rests only on calls made by the **new model (gpt-6-luna, switched
   Sep 23 20:02)**, and those calls did worse than a plain long on the same coins.
3. **For most of "this week" the market was not negative for alts.** Only BTC was soft (−3.6% from its ~86k Sep 22
   peak). Alt breadth was 0.62–0.74 on Sep 25–27, and two dips (Sep 26 evening, Sep 27 afternoon) fully reversed.
   **Shorting would have lost on Sep 24–27** (a plain short on the coins the bot analysed: −0.85/−0.68/−0.56% net at 4h).
   The broad sell-off started at **~02:45 UTC Sep 28**, about 6 hours before the log ends.
4. **At the Sep 28 turn, five things lined up against shorts at once.** Each is explained below: the daily label lags,
   the prompt oversells the daily gate, the confidence bars are out of the model's reach, short playbooks have no stake,
   and the model itself did not chase. BTC was also too big: one contract costs more risk than a $75 account allows.
5. **The upside you missed is small.** Catching the whole 6-hour flush at this account size was worth about **$1–2**.
   Don't loosen survival rules on one 6-hour sample. Fix the prompt's inaccurate picture, and add measurement so the
   next "why no shorts?" can be answered from a dashboard instead of a 12-agent investigation.

---

## 1. The premise, in numbers

| Window | Closes | Long / Short | Win rate | P&L $ | R |
|---|---|---|---|---|---|
| Sep 17–23 (the rally week) | 87 | 79 / 8 | 79% | **+$8.35** | +24.3R |
| Sep 24–28 | 25 | 21 / 4 | 60% | +$0.23 | −1.1R |
| Sep 26–28 | 6 | 5 / 1 | 50% | **−$0.28** | −1.4R |
| Last 7 days (Sep 22–28) | 59 | 49 / 10 | — | +$2.95 | +7.0R |

- The Sep 26–28 losses were 3 full −1R stops, all **longs** taken into intraday dips (TAO, ZEC, INJ). The one short
  (FOLKS funding carry) **won** +0.91R. Three −1R cards in a row *look* like a bad week, but together they cost $0.59.
- Positions opened per day: Sep 17–23 averaged 12.4. Then Sep 24: 12, Sep 25: 7, Sep 26: 2, Sep 27: 4, Sep 28: 1.
- The model did **not** go quiet. It made 36–57 order calls a day and ran 186 times in the 23-hour log. The calls
  stopped becoming orders.
- Data quirk: one TAO close is stored twice in `decisions` (`stop_loss_closed` + `futures_sell_triggered`). Any quick
  sum over `decisions` overstates the loss by $0.17. The bot's own `realized_closes()` already dedupes it.

## 2. What the market actually did

- The bot's market-state readings exist from Sep 25 13:05. Daily-average breadth (share of liquid perps up over 24h) was
  **0.74 / 0.68 / 0.62** on Sep 25/26/27, with a basket-median 24h change of +2.8 / +1.7 / +0.6%.
- BTC: ~86.0k (Sep 22) → 82.9k (Sep 28 07:43). BTC's **daily bias read "bullish" (ADX ~48) in 52 of 52 snapshots.**
- The real alt sell-off started at **02:45 UTC Sep 28**. By 07:48 breadth was **0.14** and the basket median −5.64%.
- Shorts that the bot's own model submitted and that passed the gates returned, at 4h: −1.45% (Sep 24), −1.77% (Sep 26),
  −2.05% (Sep 27), +0.39% (Sep 28). **Staying out of shorts was right until early Sep 28.**

## 3. Why so few trades: the long playbook got benched

- **continuation** made 71 of the 87 rally-week closes (+$8.23). Its last order was Sep 25 13:02:54. Since then its
  62 calls produced 0 orders, with messages like:
  `STAND ASIDE: futures limit PUMP-USDT buy — judged on its own continuation:long record (n=57) … net_of_cost=+0.130%, SE=0.242%, t=0.54; zero stake (unproven …)`
- **Why its edge fell below t=1.** The continuation store holds at most 150 rows per family (141 long / 9 short now,
  oldest Sep 24 04:50). All the Sep 17–23 rally evidence (t≈2) has aged out. What's left are calls made after the
  **gpt-5.6-luna → gpt-6-luna switch** (deploy Sep 23 20:02; 85 of 86 rally-week closes were gpt-5.6). On the same coins
  and days, a *plain* long made +0.71 / +0.54 / +0.42% at 4h (Sep 25/26/27), while the model's continuation longs made
  −0.41% on Sep 26. So the decay is **call quality** (new model and/or the Sep 25 prompt change), not the market turning.
- **Did the bench cost money?** No. Replaying the benched continuation longs gives −0.41% net (t −1.38), and only 2 of
  the 20 log-verified ones would even have filled. The stand-aside did its job.
- **Size also shrank.** Every order is at the 0.40 "explore" size, because every side the bot could trade is unproven.
  Planned risk per trade fell from $0.37 to $0.22. The logged `ADAPTIVE EDGE … {'long': 0.5}` line does **not** actually
  bind, because the 0.40 floor is smaller.
- **Stated confidence dropped.** After the Sep 25 prompt line "State the confidence you hold, not the one that clears
  it" (agent.py:1953), the model's stated confidence sits at p50 **0.69** (p90 0.73). It also doesn't rank calls
  (within-day ρ −0.15). This matters for shorts (§4b).

## 4. Why no shorts at the Sep 28 turn

The funnel in the log window was **20 short calls vs 53 long calls**. In the dump itself (02:00–08:42) it was
**5 short vs 12 long**. Of the 20 shorts: 10 were refused by the confidence floor, 5 by stand-aside, 1 by the BTC
minimum lot, and 4 were placed (all funding carry). **1 filled (FOLKS, +0.91R). Zero directional shorts reached the
exchange.**

```
 rally → model proposes longs, gates admit longs → store fills 141 long : 9 short
   ↓ regime turns (Sep 28 02:45)
 (a) daily label still "bullish" (completed bars only)  → prompt says counter-daily shorts are "BLOCKED … NOT optional"
       → model doesn't propose them (daily gate: 0 hard refusals, all self-censorship)
 (b) hatches need confidence 0.75 / 0.78 / 0.80        → model now states ~0.69 → 10 shorts refused at the floor
 (c) short playbooks have no stake (thin side → pooled row, mostly longs, t<1; fade 'no edge')  → ETH, RAVE stood aside
 (d) model's own anti-chase: "3.4 ATR below VWAP, RSI 28, wait for retest" → retest never came, limits never reached
 (e) prompt blocks steer to longs (fadeSetup says BUY when oversold; carry favoured KCS long) → 12 long calls in the dump
 (f) BTC: one contract $83.65 vs $0.57 risk budget → refused
```

**(a) The daily label lags, and the prompt oversells the gate.** The 1D bias uses only completed UTC daily bars
(tools.py:2671) over a ~50-bar window (tools.py:2619; the "30 days" comment at :2614 is wrong). A −5.6% day cannot
change it until the bar closes, and even then ~90–99% of coins stay "bullish" in simulation. During the dump, 22 of 23
analysed coins still read bullish daily, while 1h/15m were bearish on 74%/83%. The prompt says
*"Counter-daily trades are BLOCKED at the code level … This is NOT optional"* (agent.py:1680), and each analysis adds
*"Do NOT open counter-daily trades"* (analytics.py:999). Yet the same prompt says declared breakout/range_edge setups
pass *"on your call alone"* (agent.py:1798). The `daily_opposing` gate **never hard-refused a single call**. Its whole
effect was the model not proposing. The model even cited a "bullish daily" on exhausted coins, where no daily gate
applies at all.

**(b) The confidence bars no longer match the model.** A "hostile regime" (daily bearish **or** exhausted,
regime.py:27) raises the floor to 0.75 and cuts size to 0.6 for **both sides**. So a trend-aligned short on a
bearish-daily coin needs 0.75, while last week's trend-aligned longs needed 0.65. The reversal-short hatch needs 0.80
(config.py:631). The model now states ~0.69, and only 1 of 97 calls reached 0.80 (the ETH short, which was then stood
aside). All 13 confidence-floor refusals on record are shorts. 8 were fades on exhausted-bullish coins (a symmetric
case) and 5 were on bearish dailies (the real asymmetry). De-overlapped, these refused shorts made +0.70% at 4h (n=7),
and the trend-aligned subset made only ~+0.09%, about the fee. That is not evidence the floor is costing money yet.

**(c) Short playbooks have no stake.** A side with fewer than 20 of its own observations is judged on the pooled family
row (edge.py:329-368). continuation:short has 3 of its own observations; pooled continuation is ~95% longs at t 0.35.
Result: zero stake, which killed the ETH reversal short (conf 0.80) and a RAVE bearish-daily short. In fairness,
continuation:short's own 3 rows are **worse** (−0.40%), so the pooled rule didn't cost money here. fade_extreme:short is
"no edge" (n=48, t −1.8), but on data from Sep 2–16 with nothing fresh for 11 days. The 150-row cap is per family, not
per side, so a busy long side evicts short evidence.

**(d) The model's own anti-chase judgement.** The best short, RAVE (bearish on every timeframe), sat 3.4–3.9 ATR below
VWAP with RSI ~28 for ~4 hours. The model waited for a retest that never came. Short limits rested a median 1.3 ATR above
market, and 0 of 5 were reached within their 15-minute lease. Under the house rules, that choice is the model's to make,
and it applied the same discipline to PUMP longs.

**(e) The prompt steered it to longs in the dump.** `entryMap.fadeSetup` says "buy" at an oversold 15m RSI, and the
prompt calls it "ready-made" (agent.py:2469). The TAO×2/FIL fade longs were then refused by tf_conflict, which saved
money: those refused longs lost −1.08% at 4h. Funding carry pointed long on KCS. Some wording is long-framed ("Trade
STRENGTH", short confirmation RSI>65 vs long <40).

**(f) BTC is too big for the account.** The one volume-confirmed BTC breakdown short (03:31) needed $83.65 of notional
for one contract against a $0.57 risk budget, so it was refused (`MIN LOT FLOOR`).

**Checked and ruled out:** the BTC correlation veto (long-only, never fired), ADAPTIVE EDGE (only penalises longs and
doesn't bind), anti_fomo (symmetric; shorts get an extra hatch), tf_conflict (refused only longs, and those lost),
bench/no_chase/move_24h (never fired), the volatility quarantine and the screener (symmetric; the screener sorts by
absolute move, both sides).

## 5. What to do (short version — full plan with file:line, effort and risk in the recommendations file)

> **Status (Sep 28, evening):** W1, W2, W3, W5, M0 and the per-side half of M1 are shipped (plus an `httpx`
> requirements fix). No survival rule was loosened. See the Status block at the top of the recommendations file.

**Do now: make the model's picture accurate (no gate or stake changes)**
- **W1.** Rewrite the daily-gate prompt text to match what the code actually enforces: completed bars only, which side
  it gates, and which routes pass (rendered from config, so 0.80 isn't hard-coded). Keep the true judgement ("don't chase
  a fresh dump") labelled as judgement, not as a code block.
- **W2.** Make `fadeSetup` say when tf_conflict will refuse it, using one shared predicate.
- **W3/W4.** Symmetric long/short wording (strength ↔ weakness, RSI 65/35). Label prompt-only rules as judgement.
  Stop calling funding carry "the one mechanical edge".
- **W5.** Fix the docs: `is_hostile_regime` is side-blind, and the daily window is 50 bars, not 30.
- **W6/W7.** Show how stale the daily label is (today's move, and what the label would be if today closed now, as
  display only). Show how deep the current 15m leg has actually retraced, so the model can place limits that can fill.

**Measure (report-only: owner dashboard, Supervisor, and hourly log; never the trading prompt)**
- **M0.** Stamp the build/prompt hash on every call, so a model switch and a prompt change can be told apart next time.
- **M1.** Retain probes per (family, **side**), so a busy long side can't evict the short record.
- **M2–M5.** Record each call's stake and final outcome. Split the gate scoreboard into "at a turn" cells (e.g. daily
  gate while 1h+15m already agree with the short). Log confidence-floor values. Add one "evidence health" block.

**Only with evidence (pre-registered; decided on M-data, ≥20 days covering both up and down days)**
- **S1.** A thin side explores at 0.4 instead of inheriting the pooled stand-aside.
- **S2.** A side-aware hostile floor.
- **S3.** Re-key the reversal hatches on 1h+15m structure plus explore size, instead of confidence 0.80.
- **S4/S5.** Floor-refused calls as side evidence, and a per-side de-overlap key.

**Explicitly do NOT**
- Don't loosen the t<1 stand-aside, enlarge the 150-row cap, or scope verdicts by model.
- Don't lower the 0.75/0.78/0.80 confidence settings or touch daily_opposing on the strength of one flush.
- Don't add a breadth/"market is negative → short" rule. It would have shorted the Sep 26 and Sep 27 dips, which reversed.
- Don't reword the confidence instruction to make the bars reachable. That would re-inflate confidence and destroy the
  only measure of whether it means anything.

## 6. Open questions

- Where did "lost a lot" come from? The dashboard's short-range autoscale (see story.md Sep 23) or the KuCoin app? The
  bot's own records show −$0.28 over Sep 26–28.
- Model vs prompt: gpt-6-luna (Sep 23) and the confidence prompt (Sep 25) are confounded. M0 makes the next change
  separable. Is gpt-6-luna worse at continuation calls than gpt-5.6-luna? That can only be answered with a stamped A/B.
- Did the dailies flip after the Sep 28 bar closed? A log/memory snapshot from Sep 29–30 would test the lag estimate
  directly.
