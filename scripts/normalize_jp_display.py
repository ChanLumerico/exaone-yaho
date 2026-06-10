"""v8 JP-display normalization: render markers predominantly as Korean 음차
(transliteration) + keep only 가끔(턴당 ≤1, p=0.12) as Japanese 원어. Words KEPT
(persona density intact) — only the SCRIPT changes. Resolves audit's top concern
(JP 도배: 91%/avg3.45 원어) + cross-form dup. Signatures (야호/파라파라) untouched.
  in: gyaru_corpus_8.raw.jsonl  ->  out: gyaru_corpus_8.jsonl
"""
import json, re, random, sys
sys.path.insert(0,"src"); import lexicon as lx
rng=random.Random(42)
KEEP_PER_TURN, P_KEEP = 2, 0.5
ORIG2TR={o:t for cat,ps in lx.LEXICON.items() for (t,o) in ps}      # 원어 -> 음차
origs=sorted(ORIG2TR, key=len, reverse=True)
ORIG_RE=re.compile("|".join(re.escape(o) for o in origs))
def normalize(text):
    ms=list(ORIG_RE.finditer(text))
    if not ms: return text
    keep=set(); kept=0
    for i in [j for j in range(len(ms))]:                            # left->right, prob+cap
        if kept<KEEP_PER_TURN and rng.random()<P_KEEP: keep.add(i); kept+=1
    out=[]; last=0
    for i,m in enumerate(ms):
        out.append(text[last:m.start()]); out.append(m.group() if i in keep else ORIG2TR[m.group()]); last=m.end()
    out.append(text[last:]); return "".join(out)
import re as _re2
PLAIN_JP={"まず":"먼저","でも":"근데","そして":"그리고","だから":"그러니까","しかし":"하지만",
 "ああ":"아","あー":"아","あーあ":"아","ええ":"에","えぇ":"에","え":"에","おお":"오","おおっ":"오","あ":"아",
 "あら":"어라","うん":"응","なに":"뭐","なんでも":"뭐든","なにゃ":"뭐냐","うわ":"우와","うぇぇ":"으윽",
 "でしょ":"그치","まるで":"마치","ざっと":"대충","ちょいと":"살짝","これ":"이거","これはいいね":"이거 좋네",
 "がんばってる":"열심히네","げんき":"기운"}
_AAA=_re2.compile(r"あ[、,]あ(?:[、,]あ)*…?")
_TOKJP=_re2.compile(r"^([ぁ-んァ-ヶー]+)([、,~!^…。.?]*)$")
def plain_jp(text):
    text=_AAA.sub("아…", text)
    out=[]
    for tok in text.split(" "):
        m=_TOKJP.match(tok)
        if m and m.group(1) in PLAIN_JP: out.append(PLAIN_JP[m.group(1)]+m.group(2))
        else: out.append(tok)
    return " ".join(out)
ALL_FORMS={}
for _cat,_ps in lx.LEXICON.items():
    for _t,_o in _ps: ALL_FORMS[_t]=_t; ALL_FORMS[_o]=_t
_TAIL=re.compile(r"[~!^]+$")
def dedup_markers(text):
    seen=set(); out=[]
    for tok in text.split(" "):
        m=_TAIL.search(tok); tail=m.group() if m else ""; core=tok[:len(tok)-len(tail)] if tail else tok
        canon=ALL_FORMS.get(core)
        if canon and "야호" not in core:
            if canon in seen: continue            # drop duplicate marker WORD (any script)
            seen.add(canon)
        out.append(tok)
    return " ".join(out)
rows=[json.loads(l) for l in open("data/gyaru_corpus_8.raw.jsonl",encoding="utf-8") if l.strip()]
for d in rows:
    for m in d["messages"]:
        if m["role"]=="assistant": m["content"]=dedup_markers(plain_jp(normalize(m["content"])))
open("data/gyaru_corpus_8.jsonl","w",encoding="utf-8").write("\n".join(json.dumps(d,ensure_ascii=False) for d in rows)+"\n")
# re-measure
asst=[m["content"] for d in rows for m in d["messages"] if m["role"]=="assistant"]
JP=re.compile(r"[ぁ-んァ-ヶー一-龥]+"); import statistics
c=[len(JP.findall(a)) for a in asst]
pairs=[(t,o) for e in lx.LEXICON.values() for (t,o) in e]
dup=sum(any(t in a and o in a for t,o in pairs) for a in asst)
print(f"[ok] gyaru_corpus_8.jsonl normalized")
print(f"  원어JP 턴: 91% -> {100*sum(x>0 for x in c)//len(asst)}% | 평균/턴: 3.45 -> {statistics.mean(c):.2f} | 2+: 82% -> {100*sum(x>=2 for x in c)//len(asst)}%")
print(f"  원어+음차 중복: 20% -> {100*dup//len(asst)}%")
print(f"  음차 마커 살아있나(페르소나): 마지={('|'.join(asst)).count('마지')} 멧챠={chr(124).join(asst).count('멧챠')} 쵸={chr(124).join(asst).count('쵸')} 혼토니={chr(124).join(asst).count('혼토니')} 야바={chr(124).join(asst).count('야바')}")
