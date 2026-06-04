"""exaone-yaho — aligning EXAONE-3.5 to a gyaru persona (non-commercial, §14).

A meme on the surface, an LLM alignment pipeline underneath:
data synthesis -> SFT -> (optional) preference optimization -> eval -> deploy.

NOTE (§5.6, LOCKED): the yaho/parapara pipelines, decorator, scorers and
morphological tools in this package are OFFLINE TEACHERS used only during
Phase 1 data synthesis. They are NOT part of inference — the deployed artifact
is the fine-tuned weights alone, which internalize the policy.
"""

__version__ = "0.1.0"
