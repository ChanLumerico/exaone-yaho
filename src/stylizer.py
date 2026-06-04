"""§6 / §16.4 — S_LLM stylizer interface (drop-in, §4 design principle).

The stylizer rewrites neutral assistant turns into *ordinary* gyaru speech. It
is abstracted behind an interface so the backend (frontier API for gold anchors;
local Qwen-32B via MLX for bulk) can be swapped without touching the pipeline.

CRITICAL (§16.4 architecture rule): the stylizer produces ONLY ordinary gyaru
tone. It must NEVER generate the "{noun} 야호~" deflection or the "파라파라나
추고있어야겠다" line — those are produced downstream by the dense-gated
pipelines (§5.3/§5.4). Mixing them here causes double-firing / distribution
collapse.

Implementations live in Phase 1d. The full prompts are in §18 (Appendix D).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Stylizer(ABC):
    """Rewrites the assistant turns of a multi-turn dialogue into gyaru tone."""

    @abstractmethod
    def stylize_dialogue(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return a copy of ``messages`` with assistant turns restyled.

        User/system turns are returned unchanged; structure is preserved (§18 D.2).
        Must not introduce yaho/parapara meme phrases (§16.4).
        """
        raise NotImplementedError


class QwenMLXStylizer(Stylizer):  # Phase 1d — bulk synthesis (§16.4)
    """Local Qwen-32B-class via MLX (Apache-2.0 -> clean output license)."""

    def __init__(self, model_path: str, temperature: float = 1.0, few_shot: list | None = None):
        self.model_path = model_path
        self.temperature = temperature
        self.few_shot = few_shot or []  # meme-free gold (type ① only, §16.4)

    def stylize_dialogue(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError("Phase 1d — see §18 D.2 for the system prompt")
