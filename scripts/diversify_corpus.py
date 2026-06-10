"""Tier1 persona-safe diversification -> v8 corpus (gyaru_corpus_8.jsonl).
 (1) reframe flatten (그 자체로 충분해/어떻게든 되겠지 -> expanded authentic pool).
 (2) 한본어 marker REDISTRIBUTION (density-neutral): existing mechanical marker SLOTS
     re-sampled from EXPANDED pool (gyaru-go + basic-JP). basic-JP -> 음차 default +
     가끔(~25%) 원어; gyaru-go -> p_kata 0.5. SIGNATURES (야호/파라파라/설렁탕) untouched.
"""
import json, re, random, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT/"src")); import lexicon as lx
rng=random.Random(42); P_RESAMPLE=0.6
PK_GYARU, PK_BASIC = 0.5, 0.25                         # basic-JP leans Korean 음차 (가끔 원어)
BASIC_JP={"에에","헤에","와아","오오","으응","솟카","나루호도","혼토니","스고쿠","얏파리","다이부","마아","에또"}
REFRAMES=["넌 그 자체로 충분해","어떻게든 되겠지","실패도 그냥 썰이지 뭐","최강은 멘탈이지","남이 뭐라든 넌 너야",
 "남 시선 신경 끄고 너답게","무리 안 하는 것도 노력이야","넌 원래 잘하잖아","그런 날도 있는 거지 뭐",
 "넘어져도 폼나게 넘어진 거야","오늘의 너도 충분히 멋져","완벽 안 해도 그게 더 매력이지","신경 끄면 그게 곧 퀸이지",
 "쫄 거 없어 넌 이미 충분하거든","실수쯤이야 귀여운 거지","내일의 너가 알아서 할 거야","그 정도면 잘 살고 있는 거야",
 "비교는 노잼이야 넌 너대로 빛나","될 대로 되라지 그래도 넌 빛나","쪽팔린 거? 아무도 기억 안 해 걱정 마","그까짓 거에 흔들릴 너가 아니지"]
DOMINANT=[re.compile(r"넌?\s*그\s*자체로\s*충분(?:해|하니까|하잖아)?[~!]*"), re.compile(r"어떻게든\s*되겠지[~!]*")]
GROUPS={"interj":[(t,o) for t,o in lx.LEXICON["interjection"] if "야호" not in t],
        "emph":list(lx.LEXICON["emphasis"]),"fill":list(lx.LEXICON["filler"])}
FORM2ROLE={f:role for role,ps in GROUPS.items() for t,o in ps for f in (t,o)}
use=Counter()
def resample(role):
    ranked=sorted(GROUPS[role],key=lambda p:use[p[0]]); t,o=rng.choice(ranked[:max(3,len(ranked)//2)]); use[t]+=1
    pk=PK_BASIC if t in BASIC_JP else PK_GYARU
    return o if rng.random()<pk else t
TAIL=re.compile(r"[~!^]+$")
def redistribute(text):
    out=[]
    for tok in text.split(" "):
        m=TAIL.search(tok); tail=m.group() if m else ""; core=tok[:len(tok)-len(tail)] if tail else tok
        if core in FORM2ROLE and "야호" not in core and rng.random()<P_RESAMPLE: out.append(resample(FORM2ROLE[core])+tail)
        else: out.append(tok)
    return " ".join(out)
rows=[json.loads(l) for l in (ROOT/"data/gyaru_corpus_6.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
n_rf=0
for d in rows:
    for m in d["messages"]:
        if m["role"]!="assistant": continue
        c=m["content"]
        def rep(mt):
            global n_rf
            if rng.random()>=0.70: return mt.group(0)
            r=sorted(REFRAMES,key=lambda x:use[x]); pick=rng.choice(r[:6]); use[pick]+=1; n_rf+=1
            return " "+pick+("~" if not pick.endswith(("~","!")) else "")+" "
        for pat in DOMINANT: c=pat.sub(rep,c)
        c=re.sub(r"\s{2,}"," ",c).strip()
        m["content"]=redistribute(c)
(ROOT/"data/gyaru_corpus_8.jsonl").write_text("\n".join(json.dumps(d,ensure_ascii=False) for d in rows)+"\n",encoding="utf-8")
txt="\n".join(m["content"] for d in rows for m in d["messages"] if m["role"]=="assistant")
print(f"[ok] gyaru_corpus_8.jsonl ({len(rows)} dlg)  reframes flattened={n_rf}")
print("기본 일본어 음차 vs 원어 (가끔 원어 확인):")
for tl,og in [("에에","えー"),("혼토니","本当に"),("얏파리","やっぱり"),("나루호도","なるほど"),("마아","まあ")]:
    print(f"  {tl}/{og}: 음차 {txt.count(tl)}  원어 {txt.count(og)}")
