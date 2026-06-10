"""v12 recall-QA synthesis (fact-controlled). Plants DISTINCTIVE facts early, creates
depth via templated filler + teacher banter, then CONSTRUCTS the recall answer to contain
the exact facts in light-gyaru form (clean supervision — teacher is NOT trusted to recall).
Curriculum: depth d in {1,2,3} filler exchanges, fact-count k in {1,2,3}, some 2nd deeper query.
Output dialogues tagged profile="recall" so the surface step applies LOW marker density.

Usage: python scripts/synth_recall.py --n 8 --reps 7 --out data/_recall.jsonl   (~56 dlg)
       python scripts/synth_recall.py --n 3 --reps 1 --out data/_recall_pilot.jsonl  (pilot)
"""
import sys, json, argparse
sys.path.insert(0, "src")
from stylizer import QwenMLXStylizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", default="data/_recall.jsonl")
a = ap.parse_args()
sty = QwenMLXStylizer(seed=42)
rng = sty.rng

BANTER = ("너는 낮은 텐션의 나른하지만 당당한 갸루 말투 AI야. 친구랑 캐주얼하게 수다 중. "
 "상대 말 내용에 1~2문장으로 가볍게 반응하고 되묻기도 해. 위로·자기긍정 남발 금지. "
 "갸루어(에~/멧챠/소레나/🙄✌️) 자연스럽게. 짧게.")

def banter(msgs, mx=90):
    convo = [m for m in msgs if m["role"] in ("user", "assistant")]
    return sty._generate(BANTER, convo, max_tokens=mx).strip().split("\n")[0]

# off-topic filler user lines (templated → clean depth, no fact contamination)
FILLER = ["아 근데 요즘 날씨 진짜 왜 이래 ㅋㅋ", "맞다 나 어제 잠 개많이 잤어", "갑자기 배고프네 ㅋㅋ",
 "아 폰 배터리 거의 다 됐다", "오늘따라 왜 이렇게 노곤하지", "방금 택배 왔다 ㅋㅋ",
 "아 맞다 나 커피 마시고 싶다", "밖에 사람 개많네", "노래 추천 좀 있어?", "아 시간 진짜 빨리 간다"]

# recall-answer wrappers — {core} carries the EXACT facts; light gyaru only (readable)
WRAP = ["에~ {core} 🙄 까먹었엉~? ㅋㅋ", "아~ {core} 방금 말했잖아~ ✌", "그거~ {core} 이미 정해놨잖아~ 🙄",
 "에또~ {core} 맞지~? 🫶", "엥 {core} 기억 안 났어~? ㅋㅋ", "아 그거~ {core} ㅎㅎ 내가 기억해줄게~",
 "{core} 라고 했잖아~ 🙄 ㅋㅋ", "에~ {core} 그거지~ ✌"]

# scenarios: facts (distinctive pools), plant template, recall query, fact-core template, keys
SCN = [
 {"name":"travel","f":{"person":["수진","민호","지은","현우","유나"],"place":["부산","강릉","제주","여수","속초"],"date":["다음주 금요일","이번 토요일","담주 화요일","25일","다음달 초"]},
  "plant":"나 {date}에 {person}이랑 {place} 여행 가기로 했어 ㅎㅎ","query":"아 맞다 나 아까 누구랑 어디 간다고 했지? 까먹었어",
  "core":"{place} 가는 거~ {person}이랑~","keys":["place","person"]},
 {"name":"exam","f":{"subject":["정보처리기사","토익","한국사능력검정","컴활 1급","간호사 국가고시"],"nth":["세번째","두번째","네번째","첫번째","다섯번째"],"date":["다음달 15일","담주 월요일","이번주 토요일","27일","다음달 초"]},
  "plant":"나 {date}에 {subject} 시험 봐. 이번이 {nth} 도전이야 ㅠㅠ","query":"야 내가 무슨 시험 본다고 했지? 그리고 몇번째라고 했더라?",
  "core":"{subject}~ {nth} 도전~","keys":["subject","nth"]},
 {"name":"visit","f":{"who":["부모님","여자친구","대학 친구들","사촌 동생","회사 동료들"],"dish":["갈비찜","파스타","불고기","김치찜","제육볶음"]},
  "plant":"이번 주말에 {who} 집에 와서 {dish} 해주려고 ㅎㅎ","query":"ㅋㅋ 야 내가 누구 온다고 했었지? 뭐 해준다 했지?",
  "core":"{who} 오잖아~ {dish} 해준다며~","keys":["who","dish"]},
 {"name":"pet","f":{"animal":["고양이","강아지","햄스터","앵무새"],"petname":["나비","콩이","뭉치","초코","두부"]},
  "plant":"나 {animal} 한 마리 입양했어! 이름은 {petname}야 ㅎㅎ","query":"아 맞다 나 무슨 동물 키운다 했지? 이름이 뭐랬더라?",
  "core":"{animal}~ 이름 {petname}~","keys":["animal","petname"]},
 {"name":"buy","f":{"item":["노트북","에어팟","운동화","가디건","키보드"],"color":["민트색","검정","아이보리","핑크","네이비"]},
  "plant":"나 {color} {item} 샀어! 개기대돼 ㅎㅎ","query":"내가 뭐 샀다고 했지? 색깔이 뭐랬어?",
  "core":"{color} {item}~","keys":["color","item"]},
 {"name":"hobby","f":{"act":["클라이밍","베이킹","요가","수영","드로잉"],"day":["매주 수요일","주말마다","화목","금요일 저녁","일요일 아침"]},
  "plant":"나 {day}에 {act} 시작했어 ㅎㅎ 재밌어","query":"아 나 무슨 운동? 취미? 시작했다 했지? 언제 한다 했더라?",
  "core":"{act}~ {day}에~","keys":["act","day"]},
 {"name":"work","f":{"company":["판교 스타트업","외국계 회사","공기업","디자인 회사","게임 회사"],"role":["백엔드 개발","마케팅","회계","UX 디자인","기획"]},
  "plant":"나 {company}에 {role}로 이직했어! 다음주 첫출근이야","query":"야 내가 어디로 이직했다 했지? 무슨 일 한다 했어?",
  "core":"{company}~ {role}~","keys":["company","role"]},
 {"name":"event","f":{"who":["BTS","아이유","뉴진스","세븐틴","데이식스"],"place":["고척돔","올림픽공원","KSPO돔","잠실주경기장"]},
  "plant":"나 다음달에 {who} 콘서트 가! {place}에서 해 ㅎㅎ","query":"아 나 누구 콘서트 간다 했지? 어디서 한다 했더라?",
  "core":"{who} 콘서트~ {place}에서~","keys":["who","place"]},
]

def pick(pool):
    return pool[rng.randrange(len(pool))]

out = []
for rep in range(a.reps):
    for sc in SCN[:a.n]:
        rng.seed(1000 + rep * 17 + hash(sc["name"]) % 999)
        fv = {k: pick(v) for k, v in sc["f"].items()}
        msgs = []
        # 1. plant
        plant = sc["plant"].format(**fv)
        msgs.append({"role": "user", "content": plant})
        msgs.append({"role": "assistant", "content": banter(msgs)})
        # 2. depth: d filler exchanges (curriculum: rep parity -> varied depth 1..3)
        d = 1 + (rep % 3)
        used = set()
        for _ in range(d):
            fi = rng.randrange(len(FILLER))
            while fi in used and len(used) < len(FILLER): fi = rng.randrange(len(FILLER))
            used.add(fi)
            msgs.append({"role": "user", "content": FILLER[fi]})
            msgs.append({"role": "assistant", "content": banter(msgs)})
        # 3. recall query + CONSTRUCTED answer (clean fact supervision)
        msgs.append({"role": "user", "content": sc["query"]})
        core = sc["core"].format(**fv)
        ans = WRAP[rng.randrange(len(WRAP))].format(core=core)
        msgs.append({"role": "assistant", "content": ans})
        # record the gold facts for eval/traceability
        out.append({"scenario": sc["name"], "profile": "recall", "depth": d,
                    "facts": {k: fv[k] for k in sc["keys"]}, "messages": msgs})
        print(f"  [{sc['name']} rep{rep} d{d}] facts={[fv[k] for k in sc['keys']]}", flush=True)
        open(a.out, "w", encoding="utf-8").write("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n")
print(f"[done] {len(out)} recall dialogues -> {a.out}")
