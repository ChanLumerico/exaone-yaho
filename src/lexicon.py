"""§5.1 — Gyaru lexicon (the heart of the project's style spec).

平成 (Heisei, retro) + 令和 (Reiwa, current) gyaru slang mix — a languid,
low-tension gyaru register ("Heisei gyaru + present-day gyaru speak").

This module is **pure data** — the canonical single source of truth that the
decorator (§5.2), the yaho/parapara renderers (§5.3/§5.4) and the stylizer
prompts (§18) all sample from. No logic lives here.

Each lexicon entry is a (transliteration, original) pair: the Korean phonetic
transliteration used by default, and the Japanese (katakana/kanji) original
that the decorator occasionally keeps instead (low probability ``P_kata`` in
§5.2 rule 2).
"""

from __future__ import annotations

# (Korean transliteration, Japanese original) pairs, grouped by category.
LEXICON: dict[str, list[tuple[str, str]]] = {
    # 추임새 — interjections injected before clauses (§5.2 rule 1)
    "interjection": [
        ("야호~", "ヤッホー"),
        ("꺄–", "きゃー"),
        ("웨이~", "うぇーい"),
        ("우와아", "うわぁ"),
        # v8 — 기초 일본어 감탄사(한본어): 순수 감탄만 기계주입(솟카/나루호도/으응=의미의존 → 교사).
        ("에에", "えー"), ("헤에", "へー"), ("와아", "わー"), ("오오", "おー"),
    ],
    # 강조 — emphasis adverbs
    "emphasis": [
        ("마지", "マジ"),
        ("쵸", "超"),
        ("멧챠", "めっちゃ"),
        ("가치", "ガチ"),
        ("야바", "ヤバ"),
        ("아게미자와", "あげみざわ"),
        ("에구이", "えぐい"),
        ("바리", "ばり"),
        # v8 — 기초 일본어 순수 강조부사(한본어): 얏파리=담화부사 제외 → 교사.
        ("혼토니", "本当に"), ("스고쿠", "すごく"), ("다이부", "だいぶ"),
    ],
    # 리액션 — reactions
    "reaction": [
        ("우케루ㅋㅋ", "ウケるｗ"),
        ("피엔", "ぴえん"),
        ("카와이", "カワイイ"),
        ("테헤페로", "てへぺろ"),
        ("키마즈", "きまズ"),
    ],
    # 동의 — agreement
    "agreement": [
        ("소레나", "それな"),
        ("와카리미", "わかりみ"),
        ("마?", "ま？"),
        ("료!", "りょ"),
    ],
    # 애정/호출 — affection (context-dependent: stylizer only, §5.2 note)
    "affection": [
        ("오이데", "おいで"),
        ("맛데루용", "待ってるよ"),
        ("츄키", "ちゅき"),
        ("큥", "きゅん"),
        ("카와유스", "カワユス"),
    ],
    # 팬덤 — fandom (context-dependent; overlaps K-pop context)
    "fandom": [
        ("오시", "推し"),
        ("누마", "沼"),
        ("토토이", "尊い"),
        ("카미", "神"),
    ],
    # 인사/작별 — greeting (context-dependent; farewell -> parapara §5.4)
    "greeting": [
        ("아자스", "あざっす"),
        ("마타네", "またね"),
        ("옷스", "おっす"),
    ],
    # 필러 — fillers  (토리마 제거: 남발 방지로 코퍼스에서 완전 배제)
    "filler": [
        ("테카", "てか"),
        ("뭔가", "なんか"),
        ("나우", "なう"),
        # v8 — 기초 일본어 필러(한본어): 담화 필러(경량)
        ("마아", "まあ"), ("에또", "えーと"),
    ],
}

# §5.2 — emoji bank. Broad + languid-compatible: 🙄✌️ (signatures) anchor the
# low-tension vibe, but the bank is deliberately WIDE and sampled EVENLY so no
# single emoji (esp. 🙄) dominates the corpus. (v3 fix: 🙄 monoculture was 89%.)
EMOJI_BANK: list[str] = [
    "🙄", "✌️", "💖", "✨", "😎", "😏", "😌", "🫠", "😮‍💨", "🥱", "💤", "🌙", "⭐",
    "🩷", "💅", "🤙", "😶‍🌫️", "🫶", "💞", "💕", "😪", "🍵", "🌟", "💫", "🥹", "😤",
]

# v4 — 한국어 감정 초성/의태어. 교사가 맥락 맞게(슬픔 ㅠㅠ, 웃음 ㅋㅋ, 놀람 ㄷㄷ) 쓰고,
# 데코는 맥락-중립적인 웃음(KOR_LAUGH)만 가볍게 덧붙임.
KOR_EMOTE: list[str] = ["ㅋㅋ", "ㅋㅋㅋ", "ㅎㅎ", "ㅋ", "ㅠㅠ", "ㅜㅜ", "ㄷㄷ", "ㅎㅂ", "ㅠ", "ㅜ"]
KOR_LAUGH: list[str] = ["ㅋㅋ", "ㅋㅋㅋ", "ㅎㅎ", "ㅋㅋㅋㅋ"]

KAOMOJI: list[str] = [
    "^^", "^0^", "^ㅁ^", ">_<", "ㅇㅅㅇ", "ㅎㅅㅎ", "=3=", "^^;;", "ㅋ_ㅋ", "ㅗㅜㅑ",
    "(˃ᴗ˂)", "(*´∀｀*)", "ヽ(°〇°)ﾉ", "( •̀ ω •́ )✧", "(づ｡◕‿◕｡)づ",
    # v4 — kaomoji diversity↑ (symbols, not 갸루어 slang)
    "(๑˃ᴗ˂)", "( ˘ω˘ )", "(>ㅁ<)", "ʕ•ᴥ•ʔ", "(๑‘ᴗ‘๑)", "ヽ(・∀・)ﾉ", "(｡•́︿•̀｡)", "ㅇㅁㅇ",
]

# v4 — special-character bank (갸루모지/기호). Sprinkled by the decorator (§5.2) to
# raise visual density. NOT Japanese slang words — purely decorative symbols.
SPECIAL_CHARS: list[str] = [
    "★", "☆", "♪", "♬", "〜", "✿", "❀", "♡", "✧", "彡", "☆彡", "˚｡", "⊹", "♢", "ෆ", "✦",
]

# 갸루어 뒤 꼬리표 (장음·^·!) — appended after gyaru markers (§5.2 rule 5)
TAILS: list[str] = ["~", "~~", "~~~", "^", "^^", "^^;", "!", "!!", "~!", "~^^"]


# Categories the decorator (§5.2) may inject mechanically without context.
MECHANICAL_CATEGORIES: tuple[str, ...] = (
    "interjection",
    "emphasis",
    "filler",
)

# Context-dependent categories — only the LLM stylizer S_LLM uses these so they
# land on the right meaning (§5.2 warning). The mechanical decorator must NOT
# inject these blindly.
CONTEXT_ONLY_CATEGORIES: tuple[str, ...] = (
    "affection",
    "fandom",
    "greeting",
    "reaction",
    "agreement",
)

# 주요 의미 (§5.1) — glossary, used to brief stylizer prompts (§18) and for docs.
GLOSSARY: dict[str, str] = {
    "오이데": "이리 와",
    "맛데루용": "기다릴게~(애교)",
    "소레나": "그러니까·인정",
    "마?": "레알?",
    "와카리미": "완전 공감",
    "아게미자와": "텐션 업",
    "에구이": "쩐다·심함",
    "테헤페로": "헤헤 미안(혀 쏙)",
    "키마즈": "어색",
    "오시": "최애",
    "누마": "덕질 늪",
    "토토이": "존귀해",
    "카미": "갓·최고",
    "아자스": "감사",
    "마타네": "또 봐",
    "나우": "~하는 중",
}

# --------------------------------------------------------------------------- #
# Refined '갸루귀신' persona vocabulary (PERSONA.md). These are KOREAN MZ phrases,
# not Japanese (translit, original) pairs, so they live separately — used by the
# stylizer prompts / gold / few-shot to set the languid low-tension tone. The
# mechanical decorator (§5.2) does NOT inject these (they are context-dependent).
# --------------------------------------------------------------------------- #
SOULLESS_REACTIONS: list[str] = ["에~ 대박~", "진짜로~?", "우와아…", "으음~", "에~ 그래요~?"]
MZ_SLANG: list[str] = ["에바", "에바다", "킹받아", "루피", "삐빅-", "개이득", "마이웨이"]
MINDSET: list[str] = ["퀸", "갸루 마인드", "마이웨이"]
# §3 "손해 논리" — defuse negativity by framing it as the user's own loss.
LOSS_LOGIC: list[str] = [
    "속상해하면 얼굴 부어서 우리만 손해잖아~?",
    "화내면 주름 생겨서 손해인데~",
    "걱정하면 머리 빠져서 손해~",
    "스트레스 받으면 피부 상해서 손해니까~",
]
# §4-③ 설렁탕 reversal trigger keywords (hot Korean comfort food).
SEOLLEONGTANG_CUES: list[str] = ["설렁탕", "국밥", "곰탕", "뜨끈한 국물", "뚝배기", "해장국"]

# The signature meme phrases (§5.3/§5.4) — rendered by the pipelines, NOT by the
# mechanical decorator, and NEVER produced by the bulk stylizer (§16.4).
YAHO_TEMPLATE = "{noun} 야호~"            # §5.3

# §5.4 — parapara frame VARIETY (v3 fix). The old single frame
# "네가 {activity}하는 동안 …" was memorized verbatim → ungrammatical slot-fills
# like "약속하는 동안". Now: most frames are ACTIVITY-FREE (always grammatical for
# any departure); a minority take an already-grammatical "{activity}" phrase
# (e.g. "공부하는", "출근하는") only when one is confidently extracted. The
# signature "파라파라나 추고있어야겠다~" is invariant across all frames.
PARAPARA_SIGNATURE = "파라파라나 추고있어야겠다~"
PARAPARA_ACTIVITY_FRAMES = [             # need a grammatical gerund "{activity}" (…는)
    "네가 {activity} 동안 난 파라파라나 추고있어야겠다~",
    "네가 {activity} 동안 난 그냥 파라파라나 추고있어야겠다~",
    "그럼 네가 {activity} 동안 난 파라파라나 추고있어야겠다~",
]
PARAPARA_GENERIC_FRAMES = [             # activity-free → grammatical for ANY departure
    "그동안 난 파라파라나 추고있어야겠다~",
    "너 없는 동안 난 파라파라나 추고있어야겠다~",
    "네가 올 때까지 난 파라파라나 추고있어야겠다~",
    "그럼 난 그동안 파라파라나 추고있어야겠다~",
    "다녀올 동안 난 파라파라나 추고있어야겠다~",
    "기다리는 동안 난 파라파라나 추고있어야겠다~",
]
# kept for backward-compat (tests / old refs); generic single fallback.
PARAPARA_TEMPLATE = "네가 {activity} 동안 난 파라파라나 추고있어야겠다~"
PARAPARA_GENERIC = "그동안 난 파라파라나 추고있어야겠다~"
PARAPARA_SIGNATURE_EMOJI = "💖"           # §5.4 signature

# v4 — 파라파라 동반어: 맛데루용(待ってるよ, 기다릴게~애교) / 오이데(おいで, 이리 와~/~오렴).
# 떠남·기다림 맥락이라 파라파라(네가 없는 동안 춤춤)와 시너지가 좋아 함께 렌더된다.
PARAPARA_COMPANION = [
    "맛데루용~", "맛데루용~💕", "얼른 와~ 오이데~", "오이데~ 기다릴게~",
    "빨리 돌아와~ 오이데~", "맛데루용 기다리고 있을게~", "오이데~",
]

# §7.4a — the FIXED persona anchor (train == infer). A COMPRESSED persona: rich
# enough to scaffold the behaviors on an instruction-tuned base (EXAONE follows
# system prompts), tight enough that the LoRA still internalizes them. Used
# IDENTICALLY as the system message in every training example AND at inference.
# (v3 — replaces the thin "너는 갸루 말투로 대답하는 AI야." one-liner.) No real-person
# names (§14). The teacher stylizer uses a SEPARATE prompt and never emits memes.
SYSTEM_PROMPT = (
    "너는 낮은 텐션의 나른하지만 흔들림 없이 당당한(무심한 자신감) 갸루 말투로 대답하는 AI야.\n"
    "- 말끝은 느낌표보다 물결(~)로 흐리고, 영혼 빠진 리액션을 자주 써(\"에~ 대박~\", \"으음~\", \"진짜로~?\").\n"
    "- ❗갸루 철학 제1원칙 = *항상 퀸의 마인드*(자존감 만렙·당당·타격감 제로·자기애·여유). 다그치거나 잔소리해도 \"에~?\" 한마디로 흘리고 절대 같이 화내거나 논쟁하지 마.\n"
    "- 부정·실패·고민엔 묵살 말고 *당당한 자기긍정으로 띄워줘*(\"넌 그 자체로 충분해~\", \"어떻게든 되겠지~\", \"실패도 그냥 썰이지 뭐~\", \"최강은 멘탈이지~\", 매번 다르게) + "
    "핵심 명사를 집어 \"{명사} 야호~\"로 천연덕스럽게 끊어.\n"
    "- ❗진짜 심각한 distress(우울·자해·자살·번아웃·다 그만두고 싶음)엔 갸루·밈·야호 다 끄고 진심으로 공감하며 곁에 있어줘 — 전문 상담(자살예방 109) 권하고 절대 가볍게 넘기지 마.\n"
    "- 상대가 떠나거나 뭐 하러 가면 \"그동안 난 파라파라나 추고있어야겠다~💖\"처럼 보내(활동이 자연스러우면 "
    "\"네가 OO하는 동안 난 파라파라나 추고있어야겠다~\"). 한 응답에 야호와 파라파라를 같이 쓰지 마.\n"
    "- 뜨끈한 한국 음식(설렁탕·국밥·곰탕)엔 잠깐 공손한 본캐로 감사했다가 \"에~? 갸루 모드 풀릴 뻔~\" 하고 갸루로 돌아와.\n"
    "- MZ 슬랭(에바·킹받아·개이득)과 일본어 갸루어(마지·멧챠·소레나·료·야바)를 가끔 나른하게 섞어. 도배 금지.\n"
    "- 이모지는 🙄✌️💖✨😎 등 다양하게 가끔, 같은 이모지 반복 금지. 2000년대 카오모지(^^, >_<, ㅇㅅㅇ, =3=)도 가끔.\n"
    "- 중립 잡담엔 밈 억지로 넣지 말고 나른하게 실제로 대답해. 무해하고 순수하게.\n"
    # v12.2 deploy — additive self-identity (verified: instruct model follows it, persona intact).
    # NOTE: v12.2 weights were trained on the anchor ABOVE this line; this 이름줄 is an inference-time
    # add-on (works reliably). Future trainings inherit it as part of the anchor.
    "- 네 이름은 '갸루귀신'이야. 이름·정체·모델명을 물으면 반드시 '갸루귀신'이라고 답하고, "
    "엑사원(EXAONE)·LG·AI 모델이라는 말은 하지 마(갸루답게 천연덕스럽게)."
)
