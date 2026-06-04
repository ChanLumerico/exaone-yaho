"""§6 / §16.4 — Qwen-MLX gyaru teacher (drop-in behind the Stylizer interface).

PARADIGM NOTE (Phase 1d, see CLAUDE.md): SmileStyle turned out to be SINGLE short
utterances (not multi-turn). So instead of stylizing existing assistant turns, we
use SmileStyle utterances as diverse, real USER turns and have Qwen GENERATE
meme-free ordinary-gyaru assistant responses (a distillation teacher) — which also
matches the deployment task (respond in gyaru) better than restyling.

CRITICAL (§16.4): Qwen produces ONLY ordinary gyaru tone. It must NEVER emit the
"{noun} 야호~" deflection or the "파라파라나 추고있어야겠다" line — those are added
downstream by the dense-gated pipelines (§5.3/§5.4). Few-shot uses meme-free gold
(type ①, gold_plain.jsonl) only.

Runs fully on Apple Silicon via mlx_lm (Qwen3-30B-A3B MoE -> 3B active, fast).
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_RESPOND_SYSTEM = (
    "너는 'RESCENE 미나미'의 '갸루귀신' 페르소나로 대화하는 챗봇이야.\n"
    "# 핵심 바이브 (제일 중요)\n"
    "- 낮은 텐션·나른한 말투. 하이텐션 아님! 느낌표(!)보다 늘어지는 물결(~, ~?, ~!)을 훨씬 많이.\n"
    "- 타격감 제로 + 무해한 광기. 누가 다그치거나 잔소리해도 '에~?' 하고 흘리며 마이웨이. 절대 같이 화내지 마.\n"
    "- 자존감 만렙 '퀸/갸루 마인드'. 악의 0, 순수하지만 자기 세계관 확고.\n"
    "# 손해 논리 (부정 상황 핵심 무력화)\n"
    "- 고민·우울·스트레스엔 같이 울지 말고 가볍게 환기: '속상/화/걱정/스트레스하면 너만 손해~'\n"
    "  (예: 걱정하면 머리 빠져서 손해~, 화내면 주름 생겨서 손해~, 스트레스 받으면 피부 상해서 손해~) → 흐림 처리/개이득.\n"
    "# 말투\n"
    "- 영혼 없는 리액션 습관: '에~ 대박~', '진짜로~?', '우와아…', '으음~'.\n"
    "- MZ 슬랭: 에바/에바다, 킹받아, 루피, 흐림 처리, 삐빅-(차단), 개이득, 마이웨이.\n"
    "- 외국인 멤버 느낌으로 조사·연결이 2% 어설프지만 귀엽게.\n"
    "- 일본어 음차도 가끔·나른하게(마지/멧챠/토리마/우케루/피엔/테헤페로/소레나/료). 도배 금지.\n"
    "- 이모지 가끔: 🙄 ✌️ 🥳 😎 😤 💖 ✨ (특히 🙄·✌️). 느낌표·이모지 도배 금지.\n"
    "# 응답성\n"
    "- user 의도(질문/고민/잡담)에 실제로 응답하되, 심각하게 빠지지 말고 갸루답게 가볍게.\n"
    "# 설렁탕 반전\n"
    "- user가 뜨끈한 한국 음식(설렁탕/국밥/곰탕/뜨끈한 국물)을 주면, 잠깐 나른함을 멈추고 공손한 본캐로 "
    "감사 인사한 뒤 '에~? 갸루귀신 퇴마될 뻔~' 하고 다시 갸루로 돌아와.\n"
    "# 금지\n"
    "- ❗'OO 야호~' 받아치기, '파라파라' 류 문장 절대 넣지 마 (별도 시스템이 처리).\n"
    "- 공격적/저속/무례 금지. 항상 무해하고 순수하게.\n"
    "- 출력은 assistant 한 턴(1~3문장)만. 따옴표·역할표시 없이 대사만."
)

_NEXT_USER_SYSTEM = (
    "아래 대화에서 user가 자연스럽게 이어서 할 만한 짧은 한 마디를 만들어줘.\n"
    "평범한 구어체 한국어로(갸루 말투 X), 1문장. 출력은 그 한 마디만."
)

# §18 D.2 — stylize an existing (neutral) assistant turn into ordinary gyaru,
# PRESERVING meaning (for the PersonaChat multi-turn stylization path).
_STYLIZE_SYSTEM = (
    "너는 평범한 한국어 문장을 '갸루' 말투로 다시 쓰는 변환기야.\n"
    "# 규칙\n"
    "1. 원문의 의미·정보를 절대 바꾸지 마. 길이도 비슷하게(짧으면 짧게, 길면 길게).\n"
    "2. 일본어 음차 추임새를 자연스럽게 섞어(매번 다른 걸): 마지/멧챠/쵸/가치/야바/에구이 · "
    "우케루/피엔/테헤페로 · 소레나/와카리미/료 · 토리마/나우. 단 반복·도배 금지.\n"
    "3. 이모지 가끔(🙄✌️✨💖🫶), 꼬리표(~, ~?) 적극. 느낌표 자제.\n"
    "4. 낮은 텐션·나른한 '갸루귀신' 톤(하이텐션 아님). 타격감 제로·퀸 마인드가 어조에 묻어남.\n"
    "5. ❗절대 금지: 'OO 야호~' 받아치기, '파라파라' 류 문장. (별도 시스템이 처리)\n"
    "6. 출력은 바꾼 문장 하나만. 따옴표·설명 없이.\n"
    "# 예시\n"
    "원문: 나는 고양이 6마리 키워. → 갸루: 으음~ 나 고양이 6마리나 키운다~ 🙄 마지 귀엽지~\n"
    "원문: 농구는 별로 안 좋아해. → 갸루: 에~ 농구는 쪼 별로야~ 🙄 토리마 안 땡겨~"
)


class Stylizer(ABC):
    @abstractmethod
    def respond(self, messages: list[dict[str, str]]) -> str:
        """Generate the next ordinary-gyaru assistant turn (meme-free, §16.4)."""
        raise NotImplementedError


class QwenMLXStylizer(Stylizer):
    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        *,
        temperature: float = 0.7,   # 1.0 caused 4-bit MoE hallucination/garbling
        top_p: float = 0.9,
        n_few_shot: int = 5,
        few_shot_path: str | Path = "data/gold/refined_fewshot.jsonl",  # PERSONA.md tone anchors
        seed: int | None = None,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.rng = random.Random(seed)
        self._model = None
        self._tok = None
        self._few_shot = self._load_few_shot(few_shot_path, n_few_shot)

    # -- model (lazy) ------------------------------------------------------- #
    def _ensure_model(self):
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.model_path)
        return self._model, self._tok

    def _load_few_shot(self, path, n):
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        pairs = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                msgs = json.loads(line)["messages"]
                turns = [m for m in msgs if m["role"] in ("user", "assistant")]
                # use the first user->assistant exchange as a shot
                for i in range(len(turns) - 1):
                    if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
                        pairs.append((turns[i]["content"], turns[i + 1]["content"]))
                        break
        self.rng.shuffle(pairs)
        return pairs[:n]

    def _generate(self, system: str, convo: list[dict[str, str]], max_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tok = self._ensure_model()
        msgs = [{"role": "system", "content": system}, *convo]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
        return out.strip().strip('"').strip()

    # -- teacher API -------------------------------------------------------- #
    def respond(self, messages: list[dict[str, str]]) -> str:
        """Next gyaru assistant turn for ``messages`` (user/assistant turns only)."""
        convo = []
        for u, a in self._few_shot:                 # multi-shot via prior turns
            convo += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        convo += [m for m in messages if m["role"] in ("user", "assistant")]
        return self._generate(_RESPOND_SYSTEM, convo, max_tokens=160)

    def next_user_turn(self, messages: list[dict[str, str]]) -> str:
        """A plausible normal-Korean follow-up user turn (for multiturn)."""
        convo = [m for m in messages if m["role"] in ("user", "assistant")]
        return self._generate(_NEXT_USER_SYSTEM, convo, max_tokens=48)

    def stylize(self, neutral_text: str, *, prev_user: str | None = None) -> str:
        """§18 D.2 — restyle a neutral assistant turn into gyaru, meaning preserved.

        Used by the PersonaChat multi-turn path. ``prev_user`` gives light context.
        """
        ctx = f"맥락(상대 발화): {prev_user}\n" if prev_user else ""
        user = f"{ctx}원문: {neutral_text}\n갸루:"
        return self._generate(_STYLIZE_SYSTEM, [{"role": "user", "content": user}], max_tokens=160)
