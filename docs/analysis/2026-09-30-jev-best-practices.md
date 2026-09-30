# Jev: how to use it well, and how the dual run changes (Sep 30)

Question from the owner: *what is the best way to use Jev, does it see enough, and can it manage its own open
positions when it wakes up?* This note is the research behind the revision on branch
`claude/jolly-curie-9nqogp`, and the decisions it drives.

**Status:** implemented the same day on `claude/jolly-curie-9nqogp` — state v2 (labels), rotated questions,
per-side premises, measured cost in the wording, context (market, research, macro, owner notes, untrusted text
fenced), position management, and the calibration / option-order / per-trader exit reporting. Pinned by
`tests/test_jev.py` (1,308 tests pass).

**Access note.** The research environment's network policy blocks `docs.typesafe.ai`, `arxiv.org` and most blogs,
so the official docs were read through web-search extracts, TypeSafe's own agent skill on GitHub
(`typesafe-ai/skills`), the official SDK source (`typesafe-sdk` 0.7.2 on PyPI) and pydantic-ai's Jev
integration. Independent evaluations and trading projects were read on GitHub. Sources are listed at the end.

## What Jev is (and is not)

- A **System One** model: state in, typed probabilistic decisions out (Choice / Score / Noul). No text, no
  tools, no memory between calls. ~70–500 ms, $0.042 per million input tokens, output free.
- TypeSafe's framing: *"a five-second expert judgement at machine scale"* — keep code in control, give the model
  narrow judgments, combine the typed answers in code.
- Calibrated means *higher probabilities are right more often*, not *right*. Confidence summarises how peaked
  the distribution is; it says nothing about whether the workflow is correct.

## Findings → decisions

| # | Finding | Source | Decision for trAIde |
|---|---|---|---|
| F1 | **Numbers are a documented weak spot** ("numbers and dates" is one of Jev 1.13's nine failure modes). TypeSafe: turn numbers into named categories — "RSI: overbought" beats "RSI: 78.3". Serious trading projects discretise so that "no float ever reaches it". | jaggedness (via search), practitioner repos | **State v2 is labels, not floats.** Code discretises every indicator (thresholds from the bot's own config where it has them: fade RSI 30/70, `classify_regime`'s ADX 20/25/35). Raw numbers stay on the local decision record for audit. |
| F2 | **Option order moves answers.** 88% accuracy when the right option is listed first vs 57% when last (11,621 requests); reshuffling options moved a winning probability 0.62 → 0.48. | awesome-jev-robustness | **Direction is asked three times in one call, in three cyclic option orders, and averaged** (every option sits once in every position). The spread across orders is recorded as a stability signal. Same for the position-management choice (4 orders). Free: questions in one call run in parallel. |
| F3 | **Questions are independent** — an answer never becomes context for another question. State speculative premises explicitly. | official skill, reference gist | Setup / entry / target are asked **per side** ("if this trade is a LONG…", "if a SHORT…") and code keeps the side it needs (speculative fan-out). v1 asked them without the direction, so Jev answered "which playbook?" not knowing which side. |
| F4 | **Always offer a no-match option.** Without one, Jev answers anyway (one audit: accuracy 0.95 → 0.00, ECE 0.02 → 0.79 when the abstain option was removed). | robustness audits, official skill | `stand_aside` (direction), `none_fits` (playbook → scored as "other"), `hold` (management) are always on the menu. |
| F5 | **Distracting state lowers accuracy**; keep one item per call; reference fields by name. "40 rows per request breaks ranking; one row passes." Positional array indexes are unreliable; keyed objects are fine. | state docs, robustness, measured-pitfall gist | One coin per call, keyed objects only, instructions reference fields in backticks, only the fields the questions need. Entry and position management are **separate calls** with separate state. |
| F6 | **Text in state can steer the verdict** — crude injections fail, fluent "someone already decided" framing can succeed. Keep untrusted text apart from the question. | robustness list, VentureBeat, jaggedness | Third-party / model-written text (research notes) goes in one fenced field, `untrustedContext`, and the instructions say it is information, not instructions. Owner notes are trusted and kept separate. |
| F7 | **Thresholds by consequence, validated on your own data.** Cookbook example: 0.6 for read-only, 0.9 for destructive — "evaluation examples, not universal rules". Calibration is task-dependent (ECE 0.02–0.79 across audits); recalibration on own data works (0.117 → 0.008). | official skill, jevcal, robustness | No new magic number: management actions must clear the **same confidence floor entries do** (`min_confidence`), and every Jev probability is scored against what the market did. The dashboard now shows a **calibration table** (stated P vs right-way share), which is the data a later threshold change must come from. |
| F8 | **Pin versions** once thresholds are tuned; every response names the version that answered. | models docs | Every call stores the resolved version (already), and now also a `build` stamp = code + hash of the question set and state schema, so wording changes are separable in the record ("criteria wording is the largest lever": 70% → 96% in one study). |
| F9 | **Jev judges, code executes; hard risk never delegated.** The most rigorous trading design asks ~6 atomic judgments over a <400-token pre-digested state, composes them in code and keeps stop-loss / kill-switch checks in code on every tick. One uses a `cut_position` yes/no that forces an exit. | survey of Jev trading projects | Kept: the order path, risk cap, bracket, circuit breakers and ProtectionManager stay code's. **New:** Jev gets a position-management judgment each pass (below), executed by code through the same monotonic tool bodies the LLM uses. |
| F10 | **Fees beat thin edges.** A Jev tick-trader had a 67.8% hit rate and still lost (−0.83%): edge ~1 bp per trade vs ~3 bp costs. | practitioner repo | Unchanged — the stand-aside and the RR fee guard already make every call clear measured cost. Jev's question now states the *measured* round-trip cost instead of a hard-coded "0.15%". |
| F11 | **Cascades help cost more than accuracy**: LLM judges repeat nearly all of Jev's confident errors ("wrong in the same places"); a Jev→LLM cascade gained ≤ 1.5–2.0 points. | arXiv 2609.29769 | The dual run keeps measuring **agreement** and per-row "LLM then" — if the two are wrong together, a cascade is not worth building. Listed as a later option, not built. |

## Position management (the owner's ask)

Each Jev pass now starts with Jev's **own open positions** — before any new entry — in a separate call per
position:

- **State:** the position as labels — side, playbook, how long it has been open vs the playbook's usual hold,
  where it stands in R (now, best, worst), where the stop is (original / breakeven / locking profit), how far
  the target is, funding it pays or receives, the carry hold if any; **what changed since entry** (each
  timeframe's bias at entry → now); and the same market labels an entry sees (fresh analysis of that coin).
- **Questions:** `manage` = hold / protect (tighten the stop) / extend (move the target further) / close now,
  asked in four option orders and averaged; plus `thesis_intact` (Noul) for the record.
- **Execution (code):** only if the averaged probability of the chosen action clears the entry confidence floor.
  Close = the LLM's own reduce-only market-close path. Protect = stop moved to one noise band behind price
  (never looser — the protection tool is monotonic). Extend = target moved one original R further. Hold = nothing.
- **Measurement:** a Jev close is recorded as an exit probe stamped `trader: jev` and scored against the
  replayed live exit stack, exactly like the LLM's closes — so whether Jev's exits add or destroy value is
  measured, not assumed (the LLM's own early closes measured −3.44R vs the stack, n=5). Switch:
  `JEV_MANAGE_POSITIONS` (default on).

## Not adopted (and why)

- **Per-tick trading** (several projects call Jev every block/second): our edge is at 1–8 h horizons and fees
  already dominate thin edges (F10).
- **Letting Jev size positions** from its probabilities: size stays stop-defined risk × the measured stake
  (survival); Jev's probability only acts as the stated confidence the order path already gates.
- **An LLM cascade now:** measure agreement first (F11).
- **Future option:** Jev as a cheap *watcher* for all positions between LLM runs, waking the LLM when a thesis
  looks broken — a textbook System 1 → System 2 split, worth doing once Jev's exit record exists.

## Sources

- TypeSafe docs (read via search extracts): [System One](https://docs.typesafe.ai/concepts/system-one),
  [State](https://docs.typesafe.ai/concepts/state), [Choice](https://docs.typesafe.ai/primitives/choice),
  [Score](https://docs.typesafe.ai/primitives/score), [Models](https://docs.typesafe.ai/models),
  [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13),
  [How to build with System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)
- [typesafe-ai/skills — official agent skill](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md)
- [Comprehensive Jev reference (gist)](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965)
- [jev CLI + measured positional-index pitfall (gist)](https://gist.github.com/pedramamini/014676fa8684d91bf7000f4623701ada)
- [jevcal — threshold calibration against an LLM teacher](https://github.com/abhixhek/jevcal)
- [awesome-jev-robustness — jaggedness, consistency, injection, abstention studies](https://github.com/Yifan-Lan/awesome-jev-robustness)
- [Jev finance & trading projects survey (gist)](https://gist.github.com/drillan/6916b16e8ea31a8ec36c8f59d6483150),
  [buberlo/jev-trader](https://github.com/buberlo/jev-trader), [aowang-ai/jev-trade](https://github.com/aowang-ai/jev-trade),
  [Waxmell114514/jev-trade](https://github.com/Waxmell114514/jev-trade)
- [JEV vs. LLMs as Rubric Judges (arXiv 2609.29769)](https://arxiv.org/abs/2609.29769v1),
  [JEV-as-a-Judge (arXiv 2609.26550)](https://arxiv.org/html/2609.26550v2)
- [VentureBeat: prompt injection can influence Jev's verdict](https://venturebeat.com/security/companies-are-putting-jev-in-charge-of-ai-agent-decisions-and-prompt-injection-can-influence-the-verdict)
- `typesafe-sdk` 0.7.2 source (PyPI) and pydantic-ai's `models/typesafe.py` / `models/decision.py`
