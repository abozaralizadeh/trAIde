# I Gave an AI $70 and My Exchange Keys. The Trading Was the Easy Part.

*Nine months of building an autonomous crypto trader in public — and why the hardest problem was never "make the AI trade", it was "find out what's actually true".*

---

One morning my dashboard said my trading bot had returned **+72,546,760%**.

A few weeks later, after that was fixed, the same dashboard said the account was down a gentle **−0.27%** since June. Calm. Respectable. Also wrong. The real number was **−16.5%**.

Two bugs, opposite directions, same lesson. I'll come back to both. But that's basically the whole story of this project in one picture: the AI trading was never the hard part. **Knowing the truth about what it was doing** was.

*[image: dashboard screenshot — the equity curve with the 72M% spike]*

## What I'm building

**trAIde** is an autonomous crypto trading bot running on KuCoin futures with real money. Not much money — about $70 — and that's on purpose. Paper trading lies to you. Fees don't exist, fills are perfect, and nothing hurts. With $70 on the line, every bug has a price tag, and a small account makes fees brutally honest: at this size, trading costs are the difference between a strategy and a donation.

The setup:

- A **Trading Agent** (LLM) that reads the market and places orders
- A **Research Agent** that scans ~700 perpetual contracts, reads news, and curates the watchlist
- A **Supervisor Agent** I can talk to on Telegram
- A pure-code **ProtectionManager** that manages stops and trails *outside* the LLM
- A **public dashboard** where anyone can watch it win or lose in real time: [sandboxes.live/traide](https://sandboxes.live/traide)

Everything is open source: [github.com/abozaralizadeh/trAIde](https://github.com/abozaralizadeh/trAIde)

The whole design hangs on one sentence I ended up writing into the repo's instructions in capital letters:

> **Code enforces SURVIVAL. The model owns OPPORTUNITY.**

The LLM decides *what* to trade, *which direction*, and *when to do nothing*. Code decides how much can be lost, where the stop goes, and when the whole thing gets benched. I broke this rule many times. Every time, I paid for it.

## Lesson 1: My AI had no edge — and my first measurement said it was a genius

The bot was losing. Slowly, politely, consistently. So I built a way to measure whether its direction calls actually predict anything: every time the model says "long" or "short", record the market price, then check where price is 60 and 240 minutes later. No exits, no fills, no excuses. Just: *does the call predict?*

First result: **+1.24% in 15 minutes, 92% hit rate.** I had built a money printer!

I had not. I was measuring from the *limit order price* — the discounted price the order was resting at — instead of the market price at the moment of the call. I was scoring the discount as if it were prediction. Measured correctly: **−0.007%**. Nothing.

Then a second trap: overlapping samples. The bot makes many calls on the same coin within the same hour, and they're all basically the same observation. Counting them separately gave a t-statistic of 3.66 ("significant edge!"). Decimated to independent samples: **0.51**. Noise.

So the honest picture was: no measurable edge. And here's the part that actually changed the project. The Kelly criterion says the optimal bet on a zero-edge game is **zero**. So now each strategy "family" (trend continuation, fading extremes, range edges, funding plays, breakouts) gets its own live scoreboard. A family that measures no edge after costs gets benched automatically. It keeps making *paper* calls, and the moment its numbers turn, it's back in.

That's not the code overriding the model's opportunity call. That's bet sizing. Survival.

## Lesson 2: The bot was right more often than it got paid for

Here's a number that hurt. Across 81 closed trades, the positions reached a combined **+31R of peak profit** (R = the amount risked per trade). What the account actually kept: **−4.5R**.

The trades went green. The bot just didn't keep it. The take-profit targets sat at an average of 2.25R away — and got hit **zero times in 81 trades**. Meanwhile the trailing stop was mathematically unable to lock in anything meaningful until a trade was up 1.5R, which also never happened.

The fix wasn't a clever parameter. It was tying the trailing stop to each coin's own measured noise, per trade. One definition of "noise" now serves the stop, the trail, and even the threshold for waking the LLM up — because at one point the model was being asked "what now?" every time price moved 0.5%, on trades whose stops were 2.2% away. It was re-deciding hour-long ideas every few minutes, and predictably finding reasons to fidget.

*[image: a "recently closed" card from the dashboard showing a tiny win right before a huge move]*

## Lesson 3: Every safety gate is a veto waiting to happen

Once the bleeding stopped, a new failure mode arrived: the bot stopped trading. Completely. Five separate times, five different causes:

1. A `NameError` in telemetry code silently killed the measurement feed for 2.6 days, which froze every scoreboard, which benched everything.
2. Position sizing rounded trades below the exchange's minimum order size.
3. An "overextended trend" gate had no exception path — it refused a perfectly valid setup **54 times**.
4. A "timeframe conflict" gate refused the only two qualifying trades on the entire watchlist, **11 times each**. 31 runs, zero orders.
5. One unhealthy server in a load-balanced pool returned 404 on half the AI calls. One failed call kills a whole run, so 50% call failure became 100% run failure.

And my favorite: a **16-hour outage caused by one type annotation**. I added a tool with a parameter typed `List[Dict[str, Any]]`. The agent framework compiles tool signatures into strict JSON schemas, a free-form dict isn't allowed, and the error didn't just break that tool — it broke *startup*. The bot looked alive. It polled. It published the dashboard. It placed nothing. All 563 tests were green, because no test ever built the real tool set.

The debugging habit that came out of this: **separate "is it running" from "is it trading"**, every single time. A bot that is up and a bot that is working look identical from a process list.

## Lesson 4: Build in public, because strangers find your bugs

I tweeted about a nasty one: market data could sit in the bot's state for **63 hours** and still look "live". I'd added a shelf life and felt good about it.

Someone replied:

> "63 hours of stale-but-valid state is exactly the shape that never throws… Did yours have a freshness field at all?"

It did. From the very first commit. And that was the problem: it was a *write* timestamp, not a *read* timestamp. On polls where the exchange returned nothing new, the code carried the old value forward **and stamped it with the current time**. I tested it: a 63-hour-old reading reported an age of **0 seconds**, and sailed straight through the fix I had just shipped.

One reply from a stranger found a hole my own fix didn't cover. That's worth more than a hundred likes.

## Lesson 5: The AI that debugs the AI is also wrong sometimes

I don't build this alone. My pair-programmer is an AI coding assistant (Claude Code), and most of the forensic work — replaying trades against real candles, reconstructing decisions, writing the tests — happens in long sessions with it. It's genuinely great at this. It has also been confidently wrong, and the project got better the moment we both started treating *its* findings like the bot's trades: measured, not trusted.

Three real ones:

**The column-order bug.** KuCoin *spot* candles come as `[time, open, close, high, low]`. *Futures* candles come as `[time, open, high, low, close]`. The live bot handled this correctly. The offline analysis scripts didn't — they read the low as the high. One conclusion built on that ("the model's early exits cost ~2.6R") completely **reversed** when re-run correctly: the early exits had actually *helped*. Worse, the wrong conclusion had already been written into the live prompt, telling the model to stop doing something that was working.

**The scoreboard that blamed the wrong suspect.** A panel measuring "does the model close trades too early?" read *"closes destroy value: −12.4R."* When each close was cross-checked against the logs: **16 of them were the code's own trailing stop. One was the model.** And the model's one close had helped.

**The test suite that traded.** The web entrypoint starts the live trading loop on import. A test imports every module to catch broken commits. So for weeks, *every test run briefly started a live trading loop against the real account.* Nobody noticed, because the log line was swallowed by the test runner. We only caught it by counting threads.

What actually protects against a confident, fluent, wrong analysis — human or AI:

- **A replay must reproduce reality before its "what if" means anything.** One replay claimed the current rules made +16R when the real account made +8R. That gap was the alarm.
- **Mutation-test every fix**: break the code on purpose and confirm a test fails. Several of my "passing" tests turned out to prove nothing.
- **Read the raw source, not the sanitized one.** That −0.27% dashboard number? The clean published data looked perfect. The raw table underneath still held the corrupt rows — and an earlier "heal" had quietly re-anchored the index to 100, erasing four months of real losses from the chart.

## The part where it finally worked (a little)

In mid-September the bot had its first properly good stretch: **28 closed trades, 79% winners, +9.6R**, and the account climbed from about $67 to over $70 during a broad market rally.

Two humbling footnotes, because this project doesn't do clean victories:

The first big winner (+1.81R) came from the "funding carry" playbook — a strategy whose whole idea is *getting paid a funding fee for holding*. It closed at its target in 90 minutes and collected **exactly zero funding**. Right trade, wrong reason. The real signal turned out to be crowded positioning, not the fee.

And when I reconstructed the rally against real candles, the bot had caught a fraction of it. The trend strategy had been benched *during the trend*, because it was being judged on a 60-minute clock while its trades are held for about 160 minutes. Those 143 blocked calls would have returned **+1.27% after costs, 76% winners**.

My instinct was to tune everything to the rally. I actually shipped one such change — then reverted it the same day, after writing down the rule that now governs the project:

> **Decisions must follow the market.** Never ship a constant that was tuned on the current regime. If the evidence comes from one kind of market, the change must adapt on its own — or it doesn't ship.

The change I reverted had helped exactly 1 trade out of 20. In the choppy market a month earlier, the same idea would have cost ~35R.

## Where it honestly stands

Not finished. Not profitable overall. All-time: **152 closed trades, about −10R, roughly −$3** on a ~$70 account, and the (now honest) equity curve is still down double digits since June. The recent stretch is encouraging; it's also one market regime, and I've learned what one-regime evidence is worth.

What's next:

- A trailing stop that adapts to trending vs choppy markets on its own
- More samples on the one playbook that measures positive — to find out if it's real
- Keep the dashboard brutally honest, because it's the only boss this bot has

## If you're building an LLM trader, here's what I'd steal from this

1. **Measure the signal separately from the outcome.** Win rate mixes up prediction, fills, and exits. You can't fix what you can't isolate.
2. **Zero edge → zero stake.** Let measurement bench strategies, not opinions. And make sure a benched strategy can earn its way back.
3. **Every gate needs an escape hatch and a counter.** If a rule can say no, log how often it does — and what it said no to.
4. **Losing the evidence for a verdict must never look like never having had it.** Bounded buffers and sliding windows will silently un-learn things.
5. **Put the system's beliefs where strangers can see them.** A public dashboard and honest tweets have found bugs my tests didn't.
6. **Treat your AI assistant's conclusions like trades:** sized by evidence, verified against reality, and reversible.
7. **Use real money, but tiny.** $70 has taught me more than any backtest.

Watch it live, judge it harshly: **[sandboxes.live/traide](https://sandboxes.live/traide)**
Read the code, steal the ideas, tell me what I got wrong: **[github.com/abozaralizadeh/trAIde](https://github.com/abozaralizadeh/trAIde)**

---

*This is an experiment, not financial advice. The bot trades a deliberately tiny account, it has lost money overall, and nothing here is a recommendation to trade anything.*
