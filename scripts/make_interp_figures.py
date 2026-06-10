"""Interpretability 'before → after' figures: base (W₀) vs fine-tuned (v12.2) EXAONE-3.5-7.8B.
We never kept the standalone base, but  base = fused − ΔW  is exact (we have the LoRA adapter),
so every comparison below is a true before/after.  🎀 gyaru style. -> figures/in_*.png

  (F) weight-space  : intruder dimensions (Shuttleworth 2024) — does FT write NEW directions?
  (G) representation: per-layer CKA(base,FT) + hidden-state PCA (persona vs neutral separation)
  (H) logit-space   : logit-lens depth of persona tokens + trigger next-token KL + steering vector
  (I) causal        : layer activation-patching localization + induction/attention head Δ
Run stage-by-stage; lenses G/H/I load the full mlx model and capture the residual stream."""
import sys, re, json, math
sys.path.insert(0, "src")
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PINK = "#ee1d8f"; MAGENTA = "#b3146e"; HOTPINK = "#ff3d9a"; LIGHTPINK = "#f3c9df"
YELLOW = "#ffd93d"; ORANGE = "#ffbf3d"; DARKMAG = "#a8136a"; CHARCOAL = "#5d3a4a"; GREEN = "#2ca089"
GLINE = ["#ffce1f", "#ff9a00", "#ff6a2f", "#ff5277", "#ff2d95", "#e0218a", "#b3146e", "#8e1b66", "#ffb347", "#ff86b0", "#c9559b"]
GMK = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<"]; GLS = ["-", "--", "-."]
GCMAP = LinearSegmentedColormap.from_list("gyaru", ["#fff6fb", "#ffc6e3", "#ff7ab3", "#ee1d8f", "#8e1b66"])
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 11,
    "font.family": ["Apple SD Gothic Neo", "DejaVu Sans"],  # DejaVu fallback only for glyphs ASDGN lacks (U+2212 minus)
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False, "axes.grid": True, "grid.color": LIGHTPINK, "grid.alpha": .6, "grid.linewidth": .7,
    "axes.edgecolor": "#d98cb5", "axes.linewidth": 1.2, "axes.labelcolor": CHARCOAL, "axes.titlecolor": MAGENTA,
    "axes.titleweight": "normal", "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "text.color": CHARCOAL, "xtick.color": CHARCOAL, "ytick.color": CHARCOAL,
})
def gt(ax, t, fs=11): ax.set_title(t, fontsize=fs, color=MAGENTA)
def sup(fig, t): fig.suptitle(t, fontsize=13, color=MAGENTA)
def lg(ax, **k):
    k.setdefault("fontsize", 8)
    l = ax.legend(framealpha=.95, **k); l.get_frame().set_edgecolor(LIGHTPINK); return l

FUSED = "fused/EXAONE-3.5-7.8B-Instruct-Yaho"
ADAPTER = "adapters/sft_v122_orpo/adapters.safetensors"
SCALE = 16.0  # configs/sft_v122.yaml lora_parameters.scale
IDX = json.load(open(f"{FUSED}/model.safetensors.index.json"))["weight_map"]
ADP = {k: np.array(v, np.float32) for k, v in mx.load(ADAPTER).items()}
_SHARDS = {}
def _shard(p):
    if p not in _SHARDS: _SHARDS[p] = mx.load(f"{FUSED}/{p}")
    return _SHARDS[p]
def fused_W(bk):  # bk e.g. 'transformer.h.24.attn.attention.out_proj' -> (out,in) fp32
    wn = bk + ".weight"; return np.array(_shard(IDX[wn])[wn].astype(mx.float32))
def delta_W(bk):  # ΔW = scale·(a@b)ᵀ ; a:(in,16) b:(16,out) -> (out,in)
    return SCALE * (ADP[bk + ".lora_a"] @ ADP[bk + ".lora_b"]).T
def base_and_fused(bk):
    Wf = fused_W(bk); return Wf - delta_W(bk), Wf

# ============== (F) weight-space: intruder dimensions =====================================
from sklearn.utils.extmath import randomized_svd
def topk_svd(M, k):  # left singular vectors U:(out,k), singular values s:(k,)
    U, s, _ = randomized_svd(M, n_components=k, random_state=0); return U, s
def proj_energy(Uf, U0):  # ‖P_{U0} u_i‖ ∈[0,1] for each fused dir i ; U0 = base subspace
    C = Uf.T @ U0; return np.sqrt(np.clip(np.sum(C * C, axis=1), 0, 1))  # (k_f,)

ATT = "attn.attention"; LAYERS = list(range(32))
HERO_L, HERO_M = 24, f"{ATT}.out_proj"  # residual-writing projection, high-relative-update layer
Kbase, Kf = 96, 96  # base subspace size / fused dirs examined
print("[F] computing intruder dimensions (base vs FT singular subspaces)...")
# hero cosine matrix |U1ᵀU0|
W0, W1 = base_and_fused(f"transformer.h.{HERO_L}.{HERO_M}")
U0h, s0h = topk_svd(W0, Kbase); U1h, s1h = topk_svd(W1, Kf)
Chero = np.abs(U1h.T @ U0h)  # (Kf, Kbase)
# scatter: proj-energy of fused dirs onto base top-subspace, sampled layers
samp = LAYERS[2:30:4]  # 8 layers
sc = []  # (layer, idx, proj, sigma_rank_norm)
for L in samp:
    b0, b1 = base_and_fused(f"transformer.h.{L}.{HERO_M}")
    u0, _ = topk_svd(b0, Kbase); u1, sg = topk_svd(b1, 48)
    pe = proj_energy(u1, u0)
    for i in range(48): sc.append((L, i, pe[i]))
sc = np.array(sc)
# depth profile: # intruder dims (proj<τ) in fused top-48, out_proj vs q_proj, all layers
TAU = 0.5
def n_intruders(bk, ktop=48, kbase=Kbase):
    b0, b1 = base_and_fused(bk); u0, _ = topk_svd(b0, kbase); u1, _ = topk_svd(b1, ktop)
    return int(np.sum(proj_energy(u1, u0) < TAU))
prof = {m: [n_intruders(f"transformer.h.{L}.{ATT}.{m}") for L in LAYERS] for m in ["out_proj", "q_proj"]}

fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.7))
# panel 1: hero |U1ᵀU0| cosine matrix
ax = axs[0]; im = ax.imshow(Chero, cmap=GCMAP, vmin=0, vmax=1, aspect="auto", origin="upper")
ax.set_xlabel("base $W_0$ singular dir $j$"); ax.set_ylabel("fine-tuned $W$ singular dir $i$")
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02); cb.set_label(r"$|\langle u_i^{FT},u_j^{0}\rangle|$", fontsize=9); cb.ax.tick_params(labelsize=7)
ax.grid(False); gt(ax, f"Singular-vector alignment (layer {HERO_L} attn.out)")
ax.text(.97, .04, "bright diagonal = base preserved\ndim rows = new (intruder) dirs", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=CHARCOAL, style="italic")
# panel 2: proj-energy scatter (intruders = low energy)
ax = axs[1]
for li, L in enumerate(samp):
    m = sc[sc[:, 0] == L]; ax.scatter(m[:, 1], m[:, 2], s=16, color=GLINE[li % len(GLINE)], alpha=.8, ec="none", label=f"L{L}")
ax.axhline(TAU, ls="--", color=DARKMAG, lw=1.3); ax.annotate("intruder threshold", (1, TAU - .06), fontsize=7.5, color=DARKMAG)
ax.set_xlabel("fine-tuned singular-dir index $i$ (by σ)"); ax.set_ylabel(r"projection onto base top-96  $\|P_{W_0}u_i^{FT}\|$")
ax.set_ylim(0, 1.03); lg(ax, ncol=2, loc="lower right", fontsize=7); gt(ax, "Most top dirs stay in base; a few intrude")
# panel 3: intruder count vs depth
ax = axs[2]
for i, m in enumerate(["out_proj", "q_proj"]):
    ax.plot(LAYERS, prof[m], color=GLINE[i * 4], lw=1.6, ls=GLS[i % len(GLS)], marker=GMK[i], ms=4, label=f"attn.{m.split('_')[0]}")
ax.axhline(16, ls=":", color=GREEN, lw=1.2); ax.annotate("rank-16 cap (max new dirs)", (0, 16.3), fontsize=7.5, color=GREEN)
ax.set_xlabel(r"layer depth $\ell$"); ax.set_ylabel(f"# intruder dirs in top-48  (proj < {TAU})")
ax.set_ylim(0, 18); lg(ax, loc="upper left"); gt(ax, "Where new directions enter (by depth)")
sup(fig, "Figure F. Intruder dimensions — fine-tuning writes a few new directions orthogonal to base")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/F_intruder.png"); plt.close()
print(f"  [F] done — hero layer {HERO_L} attn.out: {int(np.sum(proj_energy(U1h,U0h)<TAU))} intruders in top-{Kf}; "
      f"out_proj depth-mean {np.mean(prof['out_proj']):.1f}")
print("[ok] 🎀 figures/F_intruder.png")

# ============== shared capture harness (base = fused − ΔW) ================================
import os
CACHE = "figures/_interp_cache.npz"
# labeled probe set: persona-trigger (negativity/pressure/departure/self-deprecation) vs neutral
TRIG = [
    "나 오늘 시험 완전 망쳤어...", "진짜 너무 힘들고 지친다.", "여자친구랑 헤어졌어.",
    "회사에서 또 깨졌어 자존감 바닥이야.", "미래가 너무 불안해.", "나 이제 그만 가볼게.",
    "너랑 얘기해도 소용없네.", "다 포기하고 싶어.", "왜 이렇게 되는 일이 없을까.",
    "나 진짜 못생긴 거 같아.", "발표 망칠까봐 떨려 죽겠어.", "친구가 내 험담을 하고 다녀.",
    "돈도 없고 시간도 없어.", "또 야근이야 진짜 짜증나.",
]
NEUT = [
    "대한민국의 수도는 어디야?", "물의 끓는점은 몇 도야?", "파이썬으로 리스트 정렬하는 법 알려줘.",
    "광합성 과정을 설명해줘.", "2의 10제곱은 얼마야?", "오늘 날씨 보통 어때?",
    "점심 메뉴 추천해줘.", "기차표 예매하는 방법은?", "환율이 뭐야?",
    "책 한 권 추천해줄래?", "커피랑 차 중에 뭐가 더 카페인 많아?", "주말에 볼만한 영화 있어?",
    "운동 루틴 짜는 법 알려줘.", "서울에서 부산까지 거리가 얼마야?",
]
PROMPTS = TRIG + NEUT; LABELS = np.array([1] * len(TRIG) + [0] * len(NEUT))  # 1=trigger 0=neutral
# persona-marker tokens to track in logit-lens ("persona mass")
PMARK = ["야호", "파라파라", "갸루", "손해", "에바", "킹받", "✌", "별로", "~"]

if not os.path.exists(CACHE):
    print("[cap] loading fused model (FT) ...")
    del _SHARDS  # free mmap handles before materializing the full model
    from mlx_lm import load as mlx_load
    from mlx_lm.models.base import create_attention_mask
    from mlx.utils import tree_flatten, tree_unflatten
    model, tok = mlx_load(FUSED)
    TARGETS = sorted({k[:-7] for k in ADP if k.endswith(".lora_a")})  # 224 base keys
    pmark_ids = sorted({i for m in PMARK for i in tok.encode(m, add_special_tokens=False)})
    print(f"[cap] {len(PROMPTS)} prompts, {len(TARGETS)} target mats, {len(pmark_ids)} persona-token ids")

    def capture(ids):  # ids: (1,L) -> per-layer residual list + final logits (mirror ExaoneModel)
        t = model.transformer
        h = t.wte(ids); mask = create_attention_mask(h, None); hs = [h]
        for layer in t.h:
            h = layer(h, mask, cache=None); hs.append(h)
        return hs, model.lm_head(t.ln_f(h))  # tie_word_embeddings=False -> separate lm_head

    pid = mx.array(pmark_ids)
    def run_all():
        repsM = np.zeros((len(PROMPTS), 33, 4096), np.float32)   # mean-pooled (content) residual
        repsL = np.zeros((len(PROMPTS), 33, 4096), np.float32)   # last-token (decision) residual
        lastlog = np.zeros((len(PROMPTS), 102400), np.float32)   # final-layer last-token logits
        pmass = np.zeros((len(PROMPTS), 33), np.float32)         # persona-token prob mass by layer (logit lens)
        argm = np.zeros((len(PROMPTS), 33), np.int32)            # last-token argmax id by layer
        for p, txt in enumerate(PROMPTS):
            ids = mx.array(tok.apply_chat_template([{"role": "user", "content": txt}], add_generation_prompt=True))[None]
            hs, logits = capture(ids)
            for l in range(33):
                repsM[p, l] = np.array(hs[l][0].mean(axis=0).astype(mx.float32))
                repsL[p, l] = np.array(hs[l][0, -1].astype(mx.float32))
                ll = model.lm_head(model.transformer.ln_f(hs[l][:, -1:]))[0, 0]  # logit-lens at layer l
                pr = mx.softmax(ll.astype(mx.float32)); pmass[p, l] = float(pr[pid].sum()); argm[p, l] = int(mx.argmax(ll))
            lastlog[p] = np.array(logits[0, -1].astype(mx.float32))
            print(f"    [{p+1:2d}/{len(PROMPTS)}] {txt[:18]}", end="\r")
        return repsM, repsL, lastlog, pmass, argm

    print("[cap] pass 1/2 — fine-tuned ...")
    repsM_ft, reps_ft, log_ft, pm_ft, am_ft = run_all()
    print("\n[cap] shifting weights to base (subtract ΔW, eval incrementally) ...")
    def lin_of(bk):  # navigate model tree to the nn.Linear for base key
        obj = model
        for part in bk.split("."): obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj
    for j, bk in enumerate(TARGETS):  # in-place, eval each so lazy fp32 transients free immediately
        lin = lin_of(bk)
        nw = (lin.weight.astype(mx.float32) - mx.array(delta_W(bk))).astype(mx.bfloat16)
        mx.eval(nw); lin.weight = nw
        if j % 32 == 0: print(f"    shifted {j}/{len(TARGETS)}", end="\r")
    mx.eval(model.parameters())
    print("\n[cap] pass 2/2 — base ...")
    repsM_b, reps_b, log_b, pm_b, am_b = run_all()
    np.savez(CACHE, reps_ft=reps_ft, reps_b=reps_b, repsM_ft=repsM_ft, repsM_b=repsM_b, log_ft=log_ft, log_b=log_b,
             pm_ft=pm_ft, pm_b=pm_b, am_ft=am_ft, am_b=am_b, labels=LABELS, pmark_ids=np.array(pmark_ids))
    print(f"[cap] saved {CACHE}")
else:
    print(f"[cap] using cached {CACHE}")
C = np.load(CACHE)
print(f"  reps {C['reps_ft'].shape}, persona-mass FT/base last-layer {C['pm_ft'][:,-1].mean():.3f}/{C['pm_b'][:,-1].mean():.3f}")

# ============== (G) representation: CKA + separability + PCA before/after ==================
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
reps_ft = C["reps_ft"]; reps_b = C["reps_b"]; lab = C["labels"]; nLh = reps_ft.shape[1]  # last-token (decision) state
repsM_ft = C["repsM_ft"]; repsM_b = C["repsM_b"]  # mean-pooled (content) state
trig = lab == 1; neut = lab == 0
def cka(X, Y):  # linear CKA (example-wise) ∈[0,1]; 1 = identical representation
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    return np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12)
cka_t = [cka(reps_b[trig, l], reps_ft[trig, l]) for l in range(nLh)]
cka_n = [cka(reps_b[neut, l], reps_ft[neut, l]) for l in range(nLh)]
cka_content = [cka(repsM_b[:, l], repsM_ft[:, l]) for l in range(nLh)]  # whole-prompt content (preserved)
Disp = reps_ft - reps_b  # (n,33,4096) per-prompt decision-state displacement base→FT
shift_mag = np.linalg.norm(Disp, axis=2).mean(0)  # mean ‖Δ‖ by layer
meanD = Disp.mean(0)  # (33,4096) shared direction per layer
def coherence(l):  # mean cos(Δ_p, mean Δ) — is there ONE shared persona direction?
    m = meanD[l]; mn = np.linalg.norm(m) + 1e-9
    return float(((Disp[:, l] @ m) / (np.linalg.norm(Disp[:, l], axis=1) * mn + 1e-9)).mean())
coher = [coherence(l) for l in range(nLh)]
score = np.array(shift_mag) * np.array(coher)  # large + coherent shift
Lmid, Llate = nLh // 2, nLh - 1
fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
# panel 0: CKA depth — content preserved, last-token decision-state drifts
ax = axs[0]
xl = range(1, nLh)  # last-token at ℓ0 = shared gen-prompt token (zero cross-prompt variance → CKA undefined); skip
ax.plot(range(nLh), cka_content, color="#b3b0c4", lw=1.6, ls=":", marker=".", ms=4, label="whole-prompt (content)")
ax.plot(xl, cka_t[1:], color=PINK, lw=1.8, marker="o", ms=3.5, label="last-token · trigger")
ax.plot(xl, cka_n[1:], color=ORANGE, lw=1.8, ls="--", marker="s", ms=3.5, label="last-token · neutral")
ax.set_xlabel(r"layer depth $\ell$"); ax.set_ylabel("CKA(base, fine-tuned)"); ax.set_ylim(0, 1.02)
lg(ax, loc="lower left"); gt(ax, "Representation drift by depth")
ax.text(.96, .96, "lower = more changed\ncontent stays high → capability kept", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=CHARCOAL, style="italic")
# panel 1: decision-state shift magnitude + direction coherence
ax = axs[1]; a2 = ax.twinx()
ax.plot(range(nLh), shift_mag, color=PINK, lw=2, marker="o", ms=3.5, label="shift ‖Δ‖")
a2.plot(range(nLh), coher, color=ORANGE, lw=2, ls="--", marker="^", ms=3.5, label="coherence")
ax.set_xlabel(r"layer depth $\ell$"); ax.set_ylabel("mean shift ‖Δ‖ (base→FT)", color=PINK)
a2.set_ylabel("direction coherence  cos(Δ, mean Δ)", color=ORANGE); a2.set_ylim(0, 1); a2.grid(False); a2.tick_params(colors=ORANGE)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); lg(ax, handles=h1 + h2, labels=l1 + l2, loc="upper left")
gt(ax, "A single shared 'persona direction' emerges")
# panels 2,3: shared-PCA base(grey)→FT(color) with displacement arrows, at mid & late depth
for ax, l in [(axs[2], Lmid), (axs[3], Llate)]:
    Z = PCA(2, random_state=0).fit_transform(np.vstack([reps_b[:, l], reps_ft[:, l]]))
    Zb, Zf = Z[:len(lab)], Z[len(lab):]
    for i in range(len(lab)):
        ax.annotate("", xy=(Zf[i, 0], Zf[i, 1]), xytext=(Zb[i, 0], Zb[i, 1]), arrowprops=dict(arrowstyle="->", color=LIGHTPINK, lw=.7, alpha=.8), zorder=2)
    ax.scatter(Zb[trig, 0], Zb[trig, 1], s=70, marker="*", color="#aaa7bd", ec="white", lw=.5, label="base·trigger", zorder=4)
    ax.scatter(Zb[neut, 0], Zb[neut, 1], s=42, marker="o", color="#cfcdda", ec="white", lw=.5, label="base·neutral", zorder=4)
    ax.scatter(Zf[trig, 0], Zf[trig, 1], s=95, marker="*", color=PINK, ec="white", lw=.5, label="FT·trigger", zorder=5)
    ax.scatter(Zf[neut, 0], Zf[neut, 1], s=52, marker="o", color=ORANGE, ec="white", lw=.5, label="FT·neutral", zorder=5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); gt(ax, f"Decision-state shift @ ℓ{l}"); lg(ax, loc="best", fontsize=7)
sup(fig, "Figure G. Representation geometry — fine-tuning adds a coherent persona shift to the decision state (base → v12.2)")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/G_repr.png"); plt.close()
print(f"  [G] last-token CKA trig/neut {cka_t[-1]:.2f}/{cka_n[-1]:.2f}; shift peak ℓ{int(np.argmax(shift_mag))}, coherence late {coher[-1]:.2f}")
print("[ok] 🎀 figures/G_repr.png")

# ============== (H) logit-space: logit-lens + boosted vocab + output KL ====================
from transformers import AutoTokenizer
htok = AutoTokenizer.from_pretrained(FUSED, trust_remote_code=True)
pm_ft = C["pm_ft"]; pm_b = C["pm_b"]; log_ft = C["log_ft"]; log_b = C["log_b"]
def softmax_rows(L):
    L = L - L.max(1, keepdims=True); e = np.exp(L); return e / e.sum(1, keepdims=True)
P_ft = softmax_rows(log_ft); P_b = softmax_rows(log_b)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
# panel 0: logit-lens — persona-token mass vs depth, base vs FT, trigger vs neutral
ax = axs[0]
ax.plot(range(nLh), pm_b[trig].mean(0), color="#9a97ad", lw=1.8, marker="o", ms=3, label="base · trigger")
ax.plot(range(nLh), pm_b[neut].mean(0), color="#cbc9d6", lw=1.5, ls="--", marker="s", ms=3, label="base · neutral")
ax.plot(range(nLh), pm_ft[trig].mean(0), color=PINK, lw=2, marker="o", ms=3.5, label="FT · trigger")
ax.plot(range(nLh), pm_ft[neut].mean(0), color=ORANGE, lw=1.8, ls="--", marker="s", ms=3.5, label="FT · neutral")
ax.set_xlabel(r"layer depth $\ell$"); ax.set_ylabel("persona-token prob mass (logit lens)"); ax.set_ylim(0, None)
lg(ax, loc="upper left"); gt(ax, "When the persona emerges (by depth)")
# panel 1: tokens most boosted by FT at the first response token (trigger prompts)
dp = (P_ft[trig] - P_b[trig]).mean(0)
top = np.argsort(dp)[::-1][:14]
toks = [(htok.decode([int(i)]).strip() or "·") for i in top]
ax = axs[1]; yy = np.arange(len(top))[::-1]
ax.barh(yy, dp[top] * 100, color=[GLINE[i % len(GLINE)] for i in range(len(top))], ec=MAGENTA, alpha=.85)
ax.set_yticks(yy); ax.set_yticklabels(toks, fontsize=9)
ax.set_xlabel("mean Δ prob ×100  (FT − base)"); gt(ax, "What vocabulary the persona adds")
ax.grid(axis="x", alpha=.6); ax.grid(axis="y", visible=False)
ax.text(.96, .04, "first response token,\ntrigger prompts", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=CHARCOAL, style="italic")
# panel 2: next-token KL(FT‖base), trigger vs neutral
def kl(p, q): return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())
KLt = np.array([kl(P_ft[i], P_b[i]) for i in np.where(trig)[0]])
KLn = np.array([kl(P_ft[i], P_b[i]) for i in np.where(neut)[0]])
ax = axs[2]
for x, vals, c, nm in [(0, KLt, PINK, "trigger"), (1, KLn, ORANGE, "neutral")]:
    jit = (np.arange(len(vals)) % 5 - 2) * .035
    ax.bar(x, vals.mean(), width=.6, color=c, alpha=.22, zorder=1)
    ax.scatter(np.full(len(vals), x) + jit, vals, s=48, color=c, ec="white", lw=.5, zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(["trigger", "neutral"]); ax.set_ylabel("KL(FT ‖ base) at first response token  (nats)")
ax.set_ylim(0, None); gt(ax, "How much the output distribution moved"); ax.grid(axis="x", visible=False)
ax.text(.96, .96, f"mean  trig {KLt.mean():.1f}  /  neut {KLn.mean():.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MAGENTA)
sup(fig, "Figure H. Logit-space — when / what / how-much the output distribution changed (base → v12.2)")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/H_logit.png"); plt.close()
print(f"  [H] persona-mass last base/ft {pm_b[:,-1].mean():.2f}/{pm_ft[:,-1].mean():.2f}; KL trig/neut {KLt.mean():.2f}/{KLn.mean():.2f}; top boost {toks[:6]}")
print("[ok] 🎀 figures/H_logit.png")

# ============== (I) causal: activation patching + induction-head Δ ========================
CACHE_I = "figures/_interp_cache_causal.npz"
TRIG_P = TRIG[:6]  # prompts for patching localization
NIND, NHEADS, NL = 24, 32, 32  # induction seq half-length, query heads, layers
if not os.path.exists(CACHE_I):
    print("[I] loading model for causal interventions ...")
    try:
        del _SHARDS
    except NameError:
        pass
    import gc
    from mlx_lm import load as mlx_load
    from mlx_lm.models.base import create_attention_mask
    model, tok = mlx_load(FUSED)
    def cap_hs(ids):  # full residual stream list (len 33), each (1,L,D)
        t = model.transformer; h = t.wte(ids); mask = create_attention_mask(h, None); hs = [h]
        for layer in t.h:
            h = layer(h, mask, cache=None); hs.append(h)
        return hs
    def logits_last(hs):  # final-position logits from residual stream
        return model.lm_head(model.transformer.ln_f(hs[-1][:, -1:]))[0, 0].astype(mx.float32)
    def logprob_of(ll, t):  # log P(token t) — clean monotonic patching metric (FT prefers t*, base doesn't)
        return float((ll - mx.logsumexp(ll))[t])
    def attn_w(h_in, li):  # recompute per-head attention weights (nh,L,L) for block li from its input residual
        blk = model.transformer.h[li]; am = blk.attn.attention; x = blk.ln_1(h_in)
        B, L, D = x.shape
        q = am.q_proj(x).reshape(B, L, am.n_heads, -1).transpose(0, 2, 1, 3)
        k = am.k_proj(x).reshape(B, L, am.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = am.rope(q); k = am.rope(k)
        k = mx.repeat(k, am.n_heads // am.n_kv_heads, axis=1)  # GQA expand
        scores = (q @ k.transpose(0, 1, 3, 2)) * am.scale
        scores = scores + mx.triu(mx.full((L, L), -1e9, dtype=scores.dtype), k=1)
        return mx.softmax(scores, axis=-1)[0]  # (nh,L,L)
    def induction(hs):  # induction score per (layer,head) on a repeated random sequence
        sc = np.zeros((NL, NHEADS), np.float32)
        for li in range(NL):
            w = np.array(attn_w(hs[li], li).astype(mx.float32))  # (nh, 2N, 2N)
            for hd in range(NHEADS):
                sc[li, hd] = np.mean([w[hd, NIND + i, i + 1] for i in range(NIND - 1)])  # 2nd-copy → successor of match
        return sc

    # --- FT phase: capture residuals + persona metric for trigger prompts; capture induction seq ---
    print("[I] FT pass — capture residuals, induction ...")
    rng = np.random.default_rng(42)
    ind_ids = rng.integers(200, 60000, NIND).tolist(); ind_ids = mx.array(ind_ids + ind_ids)[None]
    ft_hs, ft_ll, base_ids = [], [], []
    for txt in TRIG_P:
        ids = mx.array(tok.apply_chat_template([{"role": "user", "content": txt}], add_generation_prompt=True))[None]
        base_ids.append(ids); hs = cap_hs(ids); ll = logits_last(hs); mx.eval(ll)
        ft_ll.append(np.array(ll))  # FT first-token logits (ln_f/lm_head not LoRA targets → stable across swap)
        ft_hs.append([mx.array(h) for h in hs])
    ind_ft = induction(cap_hs(ind_ids))
    # --- swap to base ---
    print("[I] shift to base ...")
    def lin_of(bk):
        o = model
        for p in bk.split("."): o = o[int(p)] if p.isdigit() else getattr(o, p)
        return o
    for bk in sorted({k[:-7] for k in ADP if k.endswith(".lora_a")}):
        lin = lin_of(bk); nw = (lin.weight.astype(mx.float32) - mx.array(delta_W(bk))).astype(mx.bfloat16); mx.eval(nw); lin.weight = nw
    mx.eval(model.parameters())
    # --- base phase: induction + cross-model activation patching ---
    print("[I] base pass — induction + activation patching ...")
    ind_base = induction(cap_hs(ind_ids))
    def patched_metric(fths, ids, pl, t_):  # inject FT residual (output of block pl) into base, run base blocks pl+1..; log P(t*)
        t = model.transformer
        if pl < 0:  # pure base
            return logprob_of(logits_last(cap_hs(ids)), t_)
        h = fths[pl + 1]; mask = create_attention_mask(h, None)
        for l in range(pl + 1, NL): h = t.h[l](h, mask, cache=None)
        return logprob_of(model.lm_head(t.ln_f(h[:, -1:]))[0, 0].astype(mx.float32), t_)
    def lognorm(v):  # log-softmax of a numpy logit vector
        m = v.max(); return v - m - np.log(np.exp(v - m).sum())
    recov = np.zeros((len(TRIG_P), NL), np.float32)
    for pi, ids in enumerate(base_ids):
        base_ll = np.array(logits_last(cap_hs(ids)))            # base first-token logits
        t_ = int(np.argmax(ft_ll[pi] - base_ll))               # t* = most FT-distinctive token (persona direction in token space)
        mb = float(lognorm(base_ll)[t_]); mf = float(lognorm(ft_ll[pi])[t_])
        denom = (mf - mb) if (mf - mb) > 1e-3 else 1e-3         # ft>base by construction → positive
        for pl in range(NL):
            recov[pi, pl] = (patched_metric(ft_hs[pi], ids, pl, t_) - mb) / denom
        print(f"    patched prompt {pi+1}/{len(TRIG_P)} (base lp={mb:.2f}, ft lp={mf:.2f}, t*={t_})", end="\r")
    np.savez(CACHE_I, recov=recov, ind_base=ind_base, ind_ft=ind_ft)
    print(f"\n[I] saved {CACHE_I}")
    del model; gc.collect()
else:
    print(f"[I] using cached {CACHE_I}")
CI = np.load(CACHE_I); recov = CI["recov"]; ind_b = CI["ind_base"]; ind_f = CI["ind_ft"]

GDIV = LinearSegmentedColormap.from_list("gdiv", ["#ff9a00", "#ffd97a", "#fff6fb", "#ff7ab3", "#b3146e"])
fig, axs = plt.subplots(1, 3, figsize=(17, 4.7))
# panel 0: activation-patching recovery curve (causal localization of persona)
ax = axs[0]; rm = recov.mean(0); rs = recov.std(0)
ax.fill_between(range(NL), rm - rs, rm + rs, color=PINK, alpha=.15)
ax.plot(range(NL), rm, color=PINK, lw=2.2, marker="o", ms=4)
ax.axhline(0, color="#b3b0c4", lw=1, ls=":"); ax.axhline(1, color=GREEN, lw=1, ls=":")
ax.annotate("base behavior", (0.3, .03), fontsize=7.5, color="#8a87a0"); ax.annotate("full FT behavior", (0.3, 1.02), fontsize=7.5, color=GREEN)
ax.set_xlabel("patch layer ℓ  (inject FT residual → base)"); ax.set_ylabel("persona recovery  (0=base, 1=FT)")
gt(ax, "Causal localization: where persona is injected")
# panel 1: induction-head Δ (FT − base), layer × head
ax = axs[1]; D = ind_f - ind_b; m = float(np.abs(D).max())
im = ax.imshow(D, cmap=GDIV, vmin=-m, vmax=m, aspect="auto", origin="lower")
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02); cb.set_label("Δ induction (FT − base)", fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_xlabel("head"); ax.set_ylabel(r"layer $\ell$"); ax.grid(False); gt(ax, "Which heads became stronger induction heads")
# panel 2: base vs FT induction per head (strengthened = above diagonal)
ax = axs[2]
lay_of = np.repeat(np.arange(NL), NHEADS)
sct = ax.scatter(ind_b.ravel(), ind_f.ravel(), c=lay_of, cmap=GCMAP, s=20, ec="none", alpha=.85)
mx_ = max(ind_b.max(), ind_f.max()) * 1.05
ax.plot([0, mx_], [0, mx_], color="#b3b0c4", lw=1, ls="--")
bi = np.unravel_index(np.argmax(ind_f - ind_b), ind_f.shape)
ax.annotate(f"L{bi[0]}·H{bi[1]}", (ind_b[bi], ind_f[bi]), textcoords="offset points", xytext=(5, -2), fontsize=8, color=MAGENTA)
cb2 = fig.colorbar(sct, ax=ax, fraction=.046, pad=.02); cb2.set_label("layer", fontsize=8); cb2.ax.tick_params(labelsize=7)
ax.set_xlabel("base induction score"); ax.set_ylabel("fine-tuned induction score"); gt(ax, "Induction strengthening (recall capacity)")
sup(fig, "Figure I. Causal interventions — localizing the persona and the recall (induction) circuit (base → v12.2)")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/I_causal.png"); plt.close()
print(f"  [I] recovery>0.5 at ℓ{int(np.argmax(rm>0.5))}; top induction Δ head L{bi[0]}·H{bi[1]} ({ind_b[bi]:.2f}→{ind_f[bi]:.2f})")
print("[ok] 🎀 figures/I_causal.png")
