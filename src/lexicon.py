"""§5.1 — Gyaru lexicon (the heart of the project's style spec).

平成 (Heisei, retro) + 令和 (Reiwa, current) gyaru slang mix, reflecting the
Minami persona ("mom-generation Heisei gyaru + present-day gyaru speak").

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
        ("파리피", "パリピ"),
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
    ],
    # 리액션 — reactions
    "reaction": [
        ("우케루ㅋㅋ", "ウケるｗ"),
        ("피엔", "ぴえん"),
        ("카와이", "カワイイ"),
        ("테헤페로", "てへぺろ"),
        ("바빗타", "バビった"),
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
    # 필러 — fillers
    "filler": [
        ("테카", "てか"),
        ("뭔가", "なんか"),
        ("토리마", "とりま"),
        ("나우", "なう"),
    ],
}

EMOJI_BANK: list[str] = [
    "✨", "💖", "⭐", "🫶", "💅", "😻", "🩷", "🌟", "💞", "😽", "🔥", "💕", "🥹",
]

KAOMOJI: list[str] = [
    "^^", "^0^", "^ㅁ^", ">_<", "ㅇㅅㅇ", "ㅎㅅㅎ", "=3=", "^^;;", "ㅋ_ㅋ", "ㅗㅜㅑ",
    "(˃ᴗ˂)", "(*´∀｀*)", "ヽ(°〇°)ﾉ", "( •̀ ω •́ )✧", "(づ｡◕‿◕｡)づ",
]

# 갸루어 뒤 꼬리표 (장음·^·!) — appended after gyaru markers (§5.2 rule 5)
TAILS: list[str] = ["~", "~~", "~~~", "^", "^^", "^^;", "!", "!!", "~!", "~^^"]


# Categories the decorator (§5.2) may inject mechanically without context.
MECHANICAL_CATEGORIES: tuple[str, ...] = (
    "interjection",
    "emphasis",
    "filler",
    "agreement",
)

# Context-dependent categories — only the LLM stylizer S_LLM uses these so they
# land on the right meaning (§5.2 warning). The mechanical decorator must NOT
# inject these blindly.
CONTEXT_ONLY_CATEGORIES: tuple[str, ...] = (
    "affection",
    "fandom",
    "greeting",
    "reaction",
)

# 주요 의미 (§5.1) — glossary, used to brief stylizer prompts (§18) and for docs.
GLOSSARY: dict[str, str] = {
    "오이데": "이리 와",
    "맛데루용": "기다릴게~(애교)",
    "토리마": "일단·아무튼",
    "소레나": "그러니까·인정",
    "마?": "레알?",
    "와카리미": "완전 공감",
    "아게미자와": "텐션 업",
    "에구이": "쩐다·심함",
    "테헤페로": "헤헤 미안(혀 쏙)",
    "바빗타": "완전 놀람",
    "키마즈": "어색",
    "오시": "최애",
    "누마": "덕질 늪",
    "토토이": "존귀해",
    "카미": "갓·최고",
    "아자스": "감사",
    "마타네": "또 봐",
    "나우": "~하는 중",
}

# The signature meme phrases (§5.3/§5.4) — rendered by the pipelines, NOT by the
# mechanical decorator, and NEVER produced by the bulk stylizer (§16.4).
YAHO_TEMPLATE = "{noun} 야호~"            # §5.3
PARAPARA_TEMPLATE = "네가 {activity}하는 동안, 그럼 난 파라파라나 추고있어야겠다~"  # §5.4
PARAPARA_GENERIC = "그럼 난 파라파라나 추고있어야겠다~"  # §5.4 when activity is None
PARAPARA_SIGNATURE_EMOJI = "💖"           # §5.4 signature
