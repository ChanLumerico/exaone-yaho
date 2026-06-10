"""v11 long-context synthesis (beat-driven). Qwen plays USER following a beat list
(forces topic progression -> no affirmation loop) + stylizer.respond = gyaru reply
(full context). Long casual gyaru dialogues. Surface layers applied later."""
import sys, json, argparse
sys.path.insert(0,"src")
from stylizer import QwenMLXStylizer
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=3); ap.add_argument("--out",default="data/_longchat_pilot.jsonl")
ap.add_argument("--reps",type=int,default=1); a=ap.parse_args()
sty=QwenMLXStylizer(seed=42)
CASUAL_BANTER=("너는 낮은 텐션의 나른하지만 흔들림없이 당당한 갸루 말투 AI야. 친구랑 캐주얼하게 수다 떠는 중. "
 "❗핵심: '넌 충분해/어떻게든 되겠지/잘하고 있어' 같은 위로·자기긍정은 *진짜 힘든 고민일 때만 가끔*. 평범한 잡담엔 *절대 남발 금지*. "
 "대신 상대 말 *내용에 구체적으로 반응*하고, 되묻고, 네 취향·생각도 말하고, 가볍게 농담쳐. 매번 다른 얘기(반복 금지). "
 "갸루어(에~/멧챠/소레나/마지/🙄✌️)는 자연스럽게 섞고, 1~3문장으로.")
def gyaru_reply(sty, msgs):
    convo=[m for m in msgs if m["role"] in ("user","assistant")]
    return sty._generate(CASUAL_BANTER, convo, max_tokens=160).strip()

USER_SYS=("너는 갸루 말투 AI랑 캐주얼하게 수다 떠는 평범한 한국 20대야. 반말·짧게(1~2문장)·일상적으로 *다음 한 마디만* 써. "
 "❗중요: 이번 턴엔 반드시 이 의도를 담아 *새 얘기로 진행*해(칭찬에 계속 감사 X, 같은 말 반복 X): 「{beat}」. 설명/따옴표 없이 네 메시지만.")
# (name, [6 beats])
SCN=[
 ("퇴근수다",["오늘 회사에서 지친 일 푸념하며 인사","가볍게 볼 넷플 드라마/예능 추천 요청","추천 중 하나 고르고 왜 요즘 집중 안 되는지 털기","화제전환: 주말 뭐 할지 고민","주말 같이 할 만한 거 아이디어","슬슬 잘 준비한다며 작별"]),
 ("음식고민",["배고픈데 뭐 먹을지 고민이라 운 떼기","간단 메뉴 추천 요청","요리 똥손이라 푸념","갑자기 다이어트 떠올리며 갈등","좋아하는 디저트/맛집 얘기","먹으러 간다며 작별"]),
 ("운동작심삼일",["헬스 등록했는데 작심삼일 자책","동기부여 요청","어떻게 꾸준히 할지 물어보기","화제전환: 살찐 것 같다 외모 고민","자존감 얘기 듣고 반응","운동하러 간다며 작별"]),
 ("면접긴장",["내일 중요한 면접 긴장 토로","면접 팁/마음가짐 물어보기","자기소개 막막 푸념","합격하면 뭐 할지 신난 얘기","긴장 풀 겸 취미 얘기","준비하러 간다며 작별"]),
 ("덕질수다",["좋아하는 가수 컴백 신나서 얘기","어디가 좋은지 떠들기","콘서트 티켓팅 걱정","굿즈 살까 고민","덕질이 주는 행복","스밍 돌리러 간다며 작별"]),
 ("여행계획",["휴가 어디 갈지 고민 운 떼기","여행지 추천 요청","예산 걱정 푸념","같이 갈 사람/혼행 고민","가서 할 것 상상","설레며 마무리"]),
 ("게임수다",["새 게임 시작했다 신나서","공략/팁 물어보기","현질할까 고민","친구랑 같이 하는 얘기","밤새운 거 푸념","한판 더 하러 간다며 작별"]),
 ("반려동물",["강아지 키울까 고민 운 떼기","어떤 종이 좋을지","책임감/현실 걱정","산책·돌봄 얘기","귀여운 상상","결정 미루며 마무리"]),
 ("쇼핑고민",["옷 살까 말까 운 떼기","뭐가 어울릴지 물어보기","가격 부담 푸념","세일·꿀팁 정보","다른 거 충동구매 욕구","결정하고 마무리"]),
 ("시험공부",["시험 다가와서 막막","공부법 물어보기","집중 안 된다 푸념","카페가서 할까 고민","끝나면 뭐할지 상상","공부하러 간다며 작별"]),
 ("친구갈등",["친구랑 서먹해져서 고민","어떻게 풀지 물어보기","내 잘못인가 자책","먼저 연락할까 망설임","좋았던 추억 회상","용기내보겠다며 마무리"]),
 ("날씨좋은날",["날씨 너무 좋다 들떠서","나가서 뭐할지 고민","산책·피크닉 아이디어","옷차림 고민","계절 음식 얘기","나가러 간다며 작별"]),
 ("카페투어",["카페 가고싶다 운 떼기","분위기 좋은 곳 추천","디저트 추천 요청","작업 vs 수다 고민","인생샷 얘기","가자며 마무리"]),
 ("연애상담",["썸타는 중이라 운 떼기","고백할까 말까 고민","거절 두려움 토로","상대 마음 추측","데이트 코스 상상","용기내겠다며 마무리"]),
 ("자취살림",["자취 시작했다 얘기","살림 꿀팁 물어보기","요리 막막 푸념","청소 귀찮음 토로","인테리어 욕심","마무리"]),
 ("비오는날",["비 와서 우울하다","집에서 뭐할지 고민","따뜻한 음식 얘기","빗소리 감성 수다","나른한 기분","낮잠 자러 간다며 작별"]),
 ("새해다짐",["새해 목표 세운다 얘기","작년 회고 푸념","현실적 계획 고민","작심삼일 걱정","응원 듣고 반응","시작해보겠다며 마무리"]),
 ("영화수다",["방금 영화 봤다 감상","느낌 나누기","비슷한 거 추천 요청","배우 얘기로 전환","다음에 볼 것","마무리"]),
 ("월급탕진",["월급 들어왔는데 벌써 텅장","돈 어디 썼나 푸념","절약 팁 물어보기","그래도 사고싶은 거 고민","플렉스 vs 저축","마무리"]),
 ("불금계획",["불금인데 약속 없다 심심","뭐하고 놀지 고민","집콕 vs 외출","혼술/혼영 얘기","의외로 집이 최고 결론","마무리"]),
]
out=[]
for rep in range(a.reps):
    for name,beats in SCN[:a.n]:
        sty.rng.seed(42+rep)
        msgs=[]
        for beat in beats:
            flip=[{"role":"assistant" if m["role"]=="user" else "user","content":m["content"]} for m in msgs]
            u=sty._generate(USER_SYS.format(beat=beat), flip, max_tokens=55).split("\n")[0].strip()
            if not u or len(u)<2: continue
            msgs.append({"role":"user","content":u})
            msgs.append({"role":"assistant","content":gyaru_reply(sty,msgs)})
        out.append({"scenario":name,"messages":msgs})
        print(f"  [{name}] {len(msgs)//2}교환 done",flush=True)
        open(a.out,"w",encoding="utf-8").write("\n".join(json.dumps(d,ensure_ascii=False) for d in out)+"\n")
print(f"[done] {len(out)} dialogues")
