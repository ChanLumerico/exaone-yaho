"""v12.2 marker rebalance (B = data-level diversity fix) on corpus_12 assistant turns:
(1) THIN over-used markers via WHITESPACE-TOKENIZATION: delete standalone marker tokens for
    GROUPS above CAP (raw-count based, prob=1-CAP/freq). Cuts 아게미자와/マジ ~by half, flattens
    the top cluster, lowers density -> distinct-2 up. Token-level = no mid-word corruption.
(2) CHOBERI BOOST: prepend 초베리바/초베리굿/쟌넨 (~50% JP) on strong-sentiment turns lacking it.
Deterministic (seed 42). Openers (에~/오~/아~) & recall/long-ctx structure untouched. -> data/gyaru_corpus_12_2.jsonl"""
import json, re, random
rng = random.Random(42)
rows = [json.loads(l) for l in open("data/gyaru_corpus_12.jsonl") if l.strip()]

# marker GROUPS (KR + JP forms). Exclude ambiguous openers (에/오/아) — those stay.
GROUPS = {
 "agemizawa": ["아게미자와", "あげみざわ"], "maji": ["마지", "マジ"], "cho": ["쵸", "超"], "bari": ["바리", "ばり"],
 "yaba": ["야바", "ヤバ"], "nanka": ["뭔가", "なんか"], "egui": ["에구이", "えぐい"], "meccha": ["멧챠", "めっちゃ"],
 "gachi": ["가치", "ガチ"], "teka": ["테카", "てか"], "kya": ["꺄–", "きゃー"], "weei": ["웨이", "うぇーい"],
 "sugoku": ["스고쿠", "すごく"], "honto": ["혼토니", "本当に"], "daibu": ["다이부", "だいぶ"], "nau": ["나우", "なう"],
 "maa": ["마아", "まあ"], "eto": ["에또", "えーと"], "sorena": ["소레나", "それな"], "ryo": ["료"],
 "uwaa": ["우와아", "うわぁ"], "ee": ["에에"], "hee": ["헤에", "へー"], "waa": ["와아", "わー"], "oo": ["오오", "おー"],
}
FORM2G = {f: g for g, fs in GROUPS.items() for f in fs}
def raw_freq(text):
    return {g: sum(text.count(f) for f in fs) for g, fs in GROUPS.items()}
text_all = "".join(m["content"] for d in rows for m in d["messages"] if m["role"] == "assistant")
freq = raw_freq(text_all)
CAP = 350  # cut the top cluster (~500-744) ~to here -> flatten + density down; median ~262 stays
print(f"CAP {CAP}; pre top: agemizawa {freq['agemizawa']}, maji {freq['maji']}, cho {freq['cho']}")

_STRIP = "~,.!?…ㅋㅎ💖✌🙄✨😎"
def thin(text):
    out = []
    for tok in text.split(" "):
        base = tok.strip(_STRIP)
        g = FORM2G.get(base)
        if g and freq[g] > CAP and rng.random() < (1 - CAP / freq[g]):
            continue  # drop this standalone over-used marker token
        out.append(tok)
    return re.sub(r"\s{2,}", " ", " ".join(out)).strip()

STRONG_NEG = re.compile(r"망(?:했|쳤)|최악|개(?:짜증|빡)|너무\s*힘들|진짜\s*싫|폭망|멘붕|현타|좌절|짜증")
POS = re.compile(r"대박|완전\s*좋|최고|짱|개이득|신난|쩐다|개좋|렬루")
MILD = re.compile(r"아쉽|아깝|애매|쏘쏘|별로|아쉬워")
def W(kr, jp): return jp if rng.random() < 0.5 else kr
def choberi_intro(t):
    if any(x in t for x in ["초베리", "쵸베리", "チョベリ", "残念", "쟌넨"]): return None
    if STRONG_NEG.search(t): return f"에~?? 그거 완전 {W('초베리바','チョベリバ')}~ "
    if POS.search(t): return f"오~ 그거 {W('초베리굿','チョベリグ')}~ "
    if MILD.search(t): return f"음~ 좀 {W('쟌넨','残念')}~ "
    return None

chob_added = 0
for d in rows:
    for m in d["messages"]:
        if m["role"] != "assistant": continue
        c = thin(m["content"])
        intro = choberi_intro(c)
        if intro and rng.random() < 0.6:
            c = intro + c; chob_added += 1
        m["content"] = c

with open("data/gyaru_corpus_12_2.jsonl", "w", encoding="utf-8") as fh:
    for d in rows: fh.write(json.dumps(d, ensure_ascii=False) + "\n")

new_all = "".join(m["content"] for d in rows for m in d["messages"] if m["role"] == "assistant")
nf = raw_freq(new_all)
chob = sum(new_all.count(x) for x in ["초베리", "쵸베리", "チョベリ", "쟌넨", "残念"])
oldtot, newtot = sum(freq.values()), sum(nf.values())
top = sorted(nf.items(), key=lambda x: -x[1])[:6]
print(f"[ok] gyaru_corpus_12_2.jsonl (choberi +{chob_added} turns)")
print(f"  아게미자와: {freq['agemizawa']} -> {nf['agemizawa']} ({100*(freq['agemizawa']-nf['agemizawa'])//max(freq['agemizawa'],1)}% cut)")
print(f"  choberi total: 136 -> {chob}")
print(f"  marker total: {oldtot} -> {newtot} ({100*(oldtot-newtot)//max(oldtot,1)}% thinned)")
print(f"  top6 now: {top}  (was max ~744)")
