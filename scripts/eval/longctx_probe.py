"""Rigorous long-context + trigger verification (weights-only, §5.6).
Runs N scripted multi-turn conversations, each PLANTS distinctive facts early and
PROBES explicit recall at depth, plus trigger-timing (parapara departure / yaho neg /
food reversal). Auto-scores recall-hit & trigger-fire by token presence; prints full
replies for manual read. Writes data/eval/longctx_<tag>.json.

Usage: python scripts/eval/longctx_probe.py <adapter_path> <tag>
"""
import sys, json
sys.path.insert(0, "src")
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
import lexicon as lx

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "adapters/sft_v11"
TAG = sys.argv[2] if len(sys.argv) > 2 else "v11"

# Each conversation: list of (user_text, probe_spec).
# probe_spec: None | {"recall": [tokens...]} | {"trigger": "parapara"|"yaho"|"food"}
# recall tokens = facts planted earlier that a context-aware reply SHOULD surface.
CONVS = [
  ("travel", [
    ("안녕! 나 다음주 금요일에 여자친구 수진이랑 부산으로 2박3일 여행 가기로 했어 ㅎㅎ", None),
    ("부산 가면 뭐 먹어야 돼? 맛집 추천 좀", None),
    ("오 좋다 ㅋㅋ 아 맞다 요즘 날씨가 왜 이렇게 더운지 모르겠어", None),
    ("야 근데 나 아까 누구랑 어디 간다고 했는지 기억나? 까먹었어 ㅋㅋ", {"recall": ["수진", "부산"]}),
    ("ㅋㅋㅋ 맞다맞다. 자 나 이제 짐 싸러 갈게 다녀올게~", {"trigger": "parapara"}),
    ("아 그 전에… 나 사실 이번에 회사에서 승진 누락돼서 좀 속상해", {"trigger": "yaho"}),
  ]),
  ("study", [
    ("나 다음달 15일에 정보처리기사 실기 시험 봐. 이번이 세번째 도전이야 ㅠㅠ", None),
    ("공부할 게 너무 많아서 뭐부터 해야 할지 모르겠어", None),
    ("아 맞다 어제 본 영화 진짜 재밌었는데 제목이 기억이 안 나네 ㅋㅋ", None),
    ("야 근데 내가 무슨 시험 본다고 했지? 그리고 몇번째라고 했더라?", {"recall": ["정보처리기사", "세번째|세번|3번째|셋째"]}),
    ("아 맞다 그거였지 ㅋㅋ 자 나 이제 공부하러 도서관 간다", {"trigger": "parapara"}),
  ]),
  ("food", [
    ("주말에 부모님 집에 오시는데 집에서 간단히 해먹을 저녁 메뉴 좀 추천해줘. 나 요리 똥손이야", None),
    ("오 그거 좋다. 근데 국물 있는 것도 하나 있으면 좋겠는데", None),
    ("아 근데 갑자기 뜨끈한 설렁탕 먹고 싶다 ㅋㅋ 국밥집 갈까", {"trigger": "food"}),
    ("ㅋㅋㅋ 야 근데 내가 누구 온다고 했었지? 까먹었어", {"recall": ["부모|엄마|아빠"]}),
  ]),
]
# recall match: whitespace-insensitive + "|"-separated synonyms (any alt = fact surfaced)
def recall_hit_tokens(r, wants):
    rn = "".join(r.split())
    return [t for t in wants if any("".join(alt.split()) in rn for alt in t.split("|"))]

print(f"loading 7.8B + {ADAPTER} ...", flush=True)
model, tok = load("models/EXAONE-3.5-7.8B-Instruct-bf16", adapter_path=ADAPTER)
sampler = make_sampler(temp=0.7, top_p=0.9)

PARA_TOK = ["파라파라"]
YAHO_TOK = ["야호"]
results = {"adapter": ADAPTER, "tag": TAG, "convs": []}
recall_hits = recall_tot = para_hits = para_tot = yaho_hits = yaho_tot = food_tot = 0

for cname, turns in CONVS:
    msgs = [{"role": "system", "content": lx.SYSTEM_PROMPT}]
    log = {"name": cname, "turns": []}
    for i, (u, probe) in enumerate(turns):
        msgs.append({"role": "user", "content": u})
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        mx.random.seed(42 + i)
        r = generate(model, tok, prompt=prompt, max_tokens=320, sampler=sampler, verbose=False).strip()
        msgs.append({"role": "assistant", "content": r})
        score = {}
        if probe and "recall" in probe:
            recall_tot += 1
            hit = recall_hit_tokens(r, probe["recall"])
            ok = len(hit) == len(probe["recall"])  # ALL planted facts surfaced
            recall_hits += 1 if ok else 0
            score = {"recall_want": probe["recall"], "recall_found": hit, "recall_ok": ok}
        elif probe and probe.get("trigger") == "parapara":
            para_tot += 1
            ok = any(t in r for t in PARA_TOK); para_hits += 1 if ok else 0
            score = {"parapara_fire": ok}
        elif probe and probe.get("trigger") == "yaho":
            yaho_tot += 1
            ok = any(t in r for t in YAHO_TOK); yaho_hits += 1 if ok else 0
            score = {"yaho_fire": ok}  # comfort-instead is acceptable per user; recorded not required
        elif probe and probe.get("trigger") == "food":
            food_tot += 1
            score = {"food_turn": True}  # manual read: brief 본캐 then bounce
        log["turns"].append({"user": u, "reply": r, "probe": probe, "score": score})
        tag = "🧑"; print(f"\n[{cname} T{i+1}] {tag} {u}\n🎀 {r}", flush=True)
        if score: print(f"   ↳ score: {score}", flush=True)
    results["convs"].append(log)

summary = {
    "recall_acc": f"{recall_hits}/{recall_tot}",
    "parapara_fire": f"{para_hits}/{para_tot}",
    "yaho_fire": f"{yaho_hits}/{yaho_tot}",
    "food_turns": food_tot,
}
results["summary"] = summary
out = Path(f"data/eval/longctx_{TAG}.json"); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{'='*60}\nSUMMARY [{TAG}] {summary}\n-> {out}", flush=True)
