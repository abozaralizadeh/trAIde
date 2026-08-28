"""Run-level LLM settings shared by every agent run.

Why this exists: the model endpoint is an APIM **load balancer** that round-robins across
three independent Azure OpenAI resources (pietro / mirko / veronica), and the Responses API
is **stateful**. With the API default ``store=true`` every turn's output items come back with
ids minted by the serving resource (``fc_*`` for tool calls, ``rs_*`` for reasoning), and the
Agents SDK replays those items as the next turn's input. Sent to any other resource they are
rejected:

    400 - "The requested item was created under a different Azure OpenAI resource.
           Use the same resource that created the item to access it."

So every turn after the first had a 2-in-3 chance of failing, and the gateway's retry then
re-ran it on the next backend. Measured over two days: 302 failed backend calls, and 134 of
210 agent turns burned exactly two doomed calls before landing on the resource that happened
to own the ids.

``store=false`` makes a run portable: the conversation is carried entirely in the request, no
ids are minted, and any backend can serve any turn. That is what a round-robin pool requires.

Applied through ``RunConfig.model_settings``, which merges over each agent's own settings
(only the non-None fields override), so per-agent temperature/tool_choice are untouched.
"""
from __future__ import annotations

from agents import ModelSettings, RunConfig

# Pass to every Runner.run(...) as `run_config=STATELESS_RUN_CONFIG`.
STATELESS_RUN_CONFIG = RunConfig(model_settings=ModelSettings(store=False))
