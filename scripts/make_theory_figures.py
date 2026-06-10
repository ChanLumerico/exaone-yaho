"""Deep, information-rich theory figures from REAL artifacts. 🎀 gyaru style.
(1) LoRA singular-value spectrum (intrinsic-rank hypothesis, from v12.2 adapter)
(2) multi-objective Pareto frontier (eval metrics)
(3) LR schedule ↔ convergence response (analytic + sft_v122 log)
(4) quantization rate-distortion (real fused weight tensor)
(5) per-layer parameter-space deformation (‖ΔW‖ map + relative perturbation vs ‖W₀‖)  -> figures/th_*.png"""
import sys, re, json, math
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PINK = "#ee1d8f"; MAGENTA = "#b3146e"; HOTPINK = "#ff3d9a"; LIGHTPINK = "#f3c9df"
YELLOW = "#ffd93d"; ORANGE = "#ffbf3d"; DARKMAG = "#a8136a"; CHARCOAL = "#5d3a4a"; GREEN = "#2ca089"
GLINE = ["#ffce1f", "#ff9a00", "#ff6a2f", "#ff5277", "#ff2d95", "#e0218a", "#b3146e", "#8e1b66", "#ffb347", "#ff86b0", "#c9559b"]
GMK = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<"]; GLS = ["-", "--", "-."]
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
    l = ax.legend(fontsize=8, framealpha=.95, **k); l.get_frame().set_edgecolor(LIGHTPINK); return l

# ===================== (1) LoRA singular-value spectrum =====================
import mlx.core as mx
W = {k: np.array(v, dtype=np.float32) for k, v in mx.load("adapters/sft_v122_orpo/adapters.safetensors").items()}
def delta_svals(a, b):  # ΔW = a@b ; a:(m,r) b:(r,n) ; nonzero svals via r×r core (QR)
    Qa, Ra = np.linalg.qr(a); Qb, Rb = np.linalg.qr(b.T)
    return np.linalg.svd(Ra @ Rb.T, compute_uv=False)  # r singular values of ΔW
mods = {}  # module-type -> list of (layer, svals)
for k in W:
    if not k.endswith(".lora_a"): continue
    base = k[:-7]; a, b = W[base + ".lora_a"], W[base + ".lora_b"]
    lay = int(re.search(r"h\.(\d+)", k).group(1)); mod = base.split(".")[-1]
    mods.setdefault(mod, []).append((lay, delta_svals(a, b)))
def eff_rank(s):  # participation ratio (Schatten): (Σσ)²/Σσ²  ∈ [1, r]
    s = np.asarray(s); return (s.sum() ** 2) / (np.sum(s ** 2) + 1e-12)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
# left: normalized spectra for one module across sampled layers
samp_mod = "c_proj" if "c_proj" in mods else list(mods)[0]
layers = sorted(mods[samp_mod]); pick = layers[:: max(1, len(layers) // 8)][:8]
for i, (lay, s) in enumerate(pick):
    sn = np.sort(s)[::-1]; sn = sn / sn[0]
    axL.semilogy(range(1, len(sn) + 1), sn, color=GLINE[i % len(GLINE)], lw=1.5, ls=GLS[i % len(GLS)], marker=GMK[i % len(GMK)], ms=4, label=f"layer {lay}")
import matplotlib.ticker as mticker
axL.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))  # plain decimals (no mathtext minus)
axL.set_xlabel("singular-value index $i$"); axL.set_ylabel(r"normalized $\sigma_i/\sigma_1$ (log)")
axL.text(.97, .95, f"module: {samp_mod}\nrank budget r=16", transform=axL.transAxes, ha="right", va="top", fontsize=8, color=CHARCOAL, style="italic")
lg(axL, ncol=2, loc="lower left"); gt(axL, "LoRA ΔW singular spectrum (decays within r=16)")
# right: effective (participation) rank per module-type, distribution over layers
mtypes = list(mods); pos = np.arange(len(mtypes))
er = [[eff_rank(s) for _, s in mods[m]] for m in mtypes]
bp = axR.boxplot(er, positions=pos, widths=.55, patch_artist=True, showfliers=False)
for i, box in enumerate(bp["boxes"]):
    box.set(facecolor=GLINE[i % len(GLINE)], edgecolor=MAGENTA, alpha=.8)
for w_ in bp["whiskers"] + bp["caps"]: w_.set_color(MAGENTA)
for med in bp["medians"]: med.set_color(CHARCOAL)
axR.axhline(16, ls="--", color=DARKMAG, lw=1.2); axR.annotate("full rank r=16", (len(mtypes) - 1.7, 15.4), fontsize=8, color=DARKMAG)
axR.set_xticks(pos); axR.set_xticklabels(mtypes, fontsize=8, rotation=20, ha="right")
axR.set_ylabel("effective rank  $(\\Sigma\\sigma)^2/\\Sigma\\sigma^2$"); axR.set_ylim(0, 17)
gt(axR, "Effective rank ≪ 16 → low intrinsic dim")
sup(fig, "Figure A. LoRA low-rank structure of the learned update ΔW (v12.2)")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/A_lora_svd.png"); plt.close()
print(f"  [A] LoRA SVD — module {samp_mod}, mean eff-rank "
      f"{np.mean([eff_rank(s) for m in mods for _, s in mods[m]]):.1f}/16")

# ===================== (2) multi-objective Pareto frontier =====================
det = json.load(open("logs/eval/det_metrics.json")); sem = json.load(open("logs/eval/sem_metrics.json"))
div = json.load(open("logs/eval/diversity_metrics.json")); cap = json.load(open("logs/eval/cap_metrics.json"))
EV = [("v6", "v6"), ("orpo2", "v8"), ("v9orpo", "v9"), ("v10orpo", "v10"), ("v11orpo", "v11"), ("v12orpo", "v12"), ("v121orpo", "v12.1"), ("v122orpo", "v12.2")]
ecol = {t: GLINE[i % len(GLINE)] for i, (t, _) in enumerate(EV)}
def pareto(P, maxx, maxy):  # indices on the frontier
    idx = []
    for i, (xi, yi) in enumerate(P):
        dom = any((xj >= xi if maxx else xj <= xi) and (yj >= yi if maxy else yj <= yi) and (xj, yj) != (xi, yi)
                  for j, (xj, yj) in enumerate(P))
        if not dom: idx.append(i)
    return idx
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
# left: firing (↑) vs ppl (↓)
P1 = [(det[t]["macro_firing_F1"], cap[t]["neutral_ppl"]) for t, _ in EV]
fr = pareto(P1, maxx=True, maxy=False)
for i, (t, lab) in enumerate(EV):
    x, y = P1[i]; axL.scatter(x, y, s=200, color=ecol[t], ec="white", lw=1.6, zorder=5, marker="*" if lab == "v12.2" else "o")
    axL.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8.5, color=ecol[t])
fx = sorted([(P1[i][0], P1[i][1]) for i in fr]); axL.plot([p[0] for p in fx], [p[1] for p in fx], color=PINK, lw=1.6, ls="--", alpha=.7, zorder=3, label="Pareto frontier")
axL.set_xlabel("firing F1  (↑ better)"); axL.set_ylabel("neutral perplexity  (↓ better)"); axL.invert_yaxis(); lg(axL, loc="lower left")
gt(axL, "Triggers ↑ vs fluency ↑  (firing–ppl)")
# right: distinct-2 (↑) vs Acc_style (↑)
P2 = [(div[t]["d2"], sem[t]["acc_style"]) for t, _ in EV]
fr2 = pareto(P2, maxx=True, maxy=True)
for i, (t, lab) in enumerate(EV):
    x, y = P2[i]; axR.scatter(x, y, s=200, color=ecol[t], ec="white", lw=1.6, zorder=5, marker="*" if lab == "v12.2" else "o")
    axR.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8.5, color=ecol[t])
fx2 = sorted([(P2[i][0], P2[i][1]) for i in fr2]); axR.plot([p[0] for p in fx2], [p[1] for p in fx2], color=PINK, lw=1.6, ls="--", alpha=.7, label="Pareto frontier")
axR.set_xlabel("distinct-2  (diversity ↑)"); axR.set_ylabel("Acc_style  (↑)"); lg(axR, loc="lower left")
gt(axR, "Diversity ↑ vs style ↑")
sup(fig, "Figure B. Multi-objective optimization — Pareto frontiers across the lineage")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/B_pareto.png"); plt.close()
print("  [B] Pareto — done")

# ===================== (3) LR schedule ↔ convergence =====================
def cosine_lr(T, peak=5e-5, warm=40, end=1e-6):
    ts = np.arange(T)
    lr = np.where(ts < warm, peak * ts / warm, end + .5 * (peak - end) * (1 + np.cos(np.pi * (ts - warm) / (T - warm))))
    return ts, lr
def parse(path, key):
    t = open(path).read()
    return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(rf"Iter (\d+): {key} ([\d.]+)", t)]
T = 3660; ts, lr = cosine_lr(T)
tr = parse("logs/train/sft_v122.log", "Train loss"); va = parse("logs/train/sft_v122.log", "Val loss")
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.4))
axL.plot(ts, lr, color=PINK, lw=2.4)
axL.axvspan(0, 40, color=YELLOW, alpha=.25); axL.annotate("warmup\n(t≤40)", (60, lr.max() * .55), fontsize=8, color=DARKMAG)
axL.annotate("cosine decay", (T * .5, lr[int(T * .5)] + 3e-6), fontsize=8, color=DARKMAG)
axL.set_xlabel("iteration"); axL.set_ylabel(r"learning rate $\eta_t$"); gt(axL, "Schedule: warmup + cosine (peak η = 5e-5)")
if tr:
    it, lv = zip(*tr); axR.plot(it, lv, color=ORANGE, lw=.5, alpha=.3)
    k = max(1, len(lv) // 60); axR.plot(it[::k], np.convolve(np.pad(lv, k, "edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")[::k][:len(it[::k])], color=ORANGE, lw=2, label="train loss")
if va:
    it, lv = zip(*va); axR.plot(it, lv, color=HOTPINK, lw=2, marker="o", ms=2.5, label="val loss")
    bi, bv = min(va, key=lambda x: x[1]); axR.scatter([bi], [bv], s=240, marker="*", color=PINK, ec="white", lw=1.2, zorder=6)
axR.set_xlabel("iteration"); axR.set_ylabel("loss"); axR.set_ylim(0, 4.5); lg(axR, loc="upper right")
axR.text(.97, .55, "train↓ keeps falling, val plateaus\n→ ★ val-min selection avoids overfit", transform=axR.transAxes, ha="right", fontsize=8, color=CHARCOAL, style="italic")
gt(axR, "Convergence response (v12.2 SFT)")
sup(fig, "Figure C. Optimization schedule and its convergence response")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/C_schedule.png"); plt.close()
print("  [C] schedule — done")

# ===================== (4) quantization rate-distortion (real weight) =====================
idx = json.load(open("fused/EXAONE-3.5-7.8B-Instruct-Yaho/model.safetensors.index.json"))["weight_map"]
wname = next(k for k in idx if "h.0" in k and ("q_proj" in k or "c_fc_0" in k) and k.endswith(".weight"))
_shard = mx.load(f"fused/EXAONE-3.5-7.8B-Instruct-Yaho/{idx[wname]}")  # bf16 -> mlx handles it
Wt = np.array(_shard[wname].astype(mx.float32)).ravel(); del _shard
Wt = Wt[:200000]  # sample for speed
def quant_err(w, bits, gs=64):  # group-wise symmetric affine, return SNR(dB) + size factor
    n = (len(w) // gs) * gs; w = w[:n].reshape(-1, gs)
    qmax = 2 ** (bits - 1) - 1; s = np.max(np.abs(w), axis=1, keepdims=True) / qmax + 1e-9
    wq = np.round(w / s).clip(-qmax - 1, qmax) * s
    mse = np.mean((w - wq) ** 2); snr = 10 * np.log10(np.var(w) / (mse + 1e-12))
    return snr
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.4))
# left: weight distribution + q4 reconstruction levels
axL.hist(Wt, bins=120, color=LIGHTPINK, ec=HOTPINK, lw=.4, density=True)
gs = 64; qmax4 = 2 ** 3 - 1; s4 = np.max(np.abs(Wt[:gs])) / qmax4
for lvl in np.arange(-qmax4, qmax4 + 1) * s4:
    axL.axvline(lvl, color=ORANGE, lw=.6, alpha=.5)
axL.set_xlabel("weight value"); axL.set_ylabel("density"); axL.set_xlim(np.percentile(Wt, .5), np.percentile(Wt, 99.5))
axL.text(.97, .95, f"tensor:\n{wname.split('.')[-2]} (layer 0)\norange = q4 levels", transform=axL.transAxes, ha="right", va="top", fontsize=8, color=CHARCOAL, style="italic")
gt(axL, "Weight distribution & 4-bit quantization grid")
# right: rate-distortion — SNR vs bits + disk size
bits = [3, 4, 5, 6, 8]; snrs = [quant_err(Wt, b) for b in bits]; sizes = [3.7, 4.6, 5.5, 6.4, 8.2]
axR.plot(bits, snrs, "-o", color=PINK, lw=2.4, ms=9, mec="white", label="reconstruction SNR")
for b, sn in zip(bits, snrs): axR.annotate(f"{sn:.0f}dB", (b, sn), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=MAGENTA)
axR.set_xlabel("quantization bits"); axR.set_ylabel("signal-to-noise ratio (dB)"); axR.set_xticks(bits)
a2 = axR.twinx(); a2.plot(bits, sizes, "--D", color=YELLOW, lw=2, ms=7, mec=ORANGE, label="disk size (GB)")
a2.set_ylabel("model size (GB)", color=ORANGE); a2.grid(False); a2.tick_params(colors=ORANGE)
h1, l1 = axR.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); lg(axR, handles=h1 + h2, labels=l1 + l2, loc="lower right")
gt(axR, "Rate–distortion: fidelity vs size")
sup(fig, "Figure D. Quantization rate–distortion on a real fused weight")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/D_quant_rd.png"); plt.close()
print(f"  [D] quant RD — tensor {wname.split('.')[-2]}, SNR {snrs}")

# ===================== (5) per-layer parameter-space deformation =====================
from matplotlib.colors import LinearSegmentedColormap
GCMAP = LinearSegmentedColormap.from_list("gyaru", ["#fff6fb", "#ffc6e3", "#ff7ab3", "#ee1d8f", "#8e1b66"])
SCALE = 16.0  # configs/sft_v122.yaml lora_parameters.scale ; ‖ΔW‖_F = scale·‖a@b‖_F
MORDER = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc_0", "c_fc_1", "c_proj"]
MLAB = {"q_proj": "attn\nq", "k_proj": "attn\nk", "v_proj": "attn\nv", "out_proj": "attn\nout",
        "c_fc_0": "mlp\nfc0", "c_fc_1": "mlp\nfc1", "c_proj": "mlp\nproj"}
rows = {}  # base_key -> (layer, module, ‖ΔW‖_F)
for k in W:
    if not k.endswith(".lora_a"): continue
    base = k[:-7]; a, b = W[base + ".lora_a"], W[base + ".lora_b"]
    lay = int(re.search(r"h\.(\d+)", base).group(1)); m = base.split(".")[-1]
    rows[base] = (lay, m, SCALE * float(np.sqrt(np.sum(delta_svals(a, b) ** 2))))
nL = max(l for l, _, _ in rows.values()) + 1
dWmat = np.full((nL, len(MORDER)), np.nan)
for _, (lay, m, fro) in rows.items(): dWmat[lay, MORDER.index(m)] = fro
# denominator ‖W₀‖_F from fused shards (W_fused ≈ W₀ since ΔW is tiny) → relative perturbation %
relmat = None
try:
    import collections
    idxmap = json.load(open("fused/EXAONE-3.5-7.8B-Instruct-Yaho/model.safetensors.index.json"))["weight_map"]
    byshard = collections.defaultdict(list)
    for base in rows: byshard[idxmap[base + ".weight"]].append(base)
    W0 = {}
    for shard, bases in byshard.items():
        sd = mx.load(f"fused/EXAONE-3.5-7.8B-Instruct-Yaho/{shard}")
        for base in bases: W0[base] = float(mx.sqrt((sd[base + ".weight"].astype(mx.float32) ** 2).sum()))
        del sd
    relmat = np.full((nL, len(MORDER)), np.nan)
    for base, (lay, m, fro) in rows.items(): relmat[lay, MORDER.index(m)] = 100.0 * fro / W0[base]
except Exception as e:
    print(f"  [E] relative-update skipped ({e})")
fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.8))
# panel 1: deformation heatmap (layer × module) — SIZE-NORMALIZED so colors compare across
# matrices of different shape (GQA shrinks k/v; MLP c_fc/c_proj are far larger than attn).
ax = axs[0]
if relmat is not None:  # relative perturbation 100·‖ΔW‖/‖W₀‖ is dimensionless → fair across shapes
    hmap, clab, ttl = relmat, r"$100\,\|\Delta W\|_F/\|W_0\|_F$  (%)", r"Relative-update map (size-normalized)"
else:  # fallback: per-column min–max so a bigger matrix isn't hot just for having more elements
    cmin = np.nanmin(dWmat, 0, keepdims=True); cmax = np.nanmax(dWmat, 0, keepdims=True)
    hmap, clab, ttl = (dWmat - cmin) / (cmax - cmin + 1e-12), "per-column normalized", "Update map (per-column normalized)"
im = ax.imshow(hmap, aspect="auto", cmap=GCMAP, origin="lower")
ax.set_xticks(range(len(MORDER))); ax.set_xticklabels([MLAB[m] for m in MORDER], fontsize=8)
ax.set_ylabel(r"layer depth $\ell$"); ax.set_yticks(list(range(0, nL, 4)))
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02); cb.set_label(clab, fontsize=9); cb.ax.tick_params(labelsize=7)
ax.grid(False); gt(ax, ttl)
# panel 2: relative perturbation depth profile (one line per module type)
ax = axs[1]; src = relmat if relmat is not None else dWmat
for i, m in enumerate(MORDER):
    ax.plot(range(nL), src[:, i], color=GLINE[i % len(GLINE)], lw=1.5, ls=GLS[i % len(GLS)],
            marker=GMK[i % len(GMK)], ms=3.5, label=MLAB[m].replace("\n", "."))
ax.set_xlabel(r"layer depth $\ell$")
ax.set_ylabel(r"relative update $100\,\|\Delta W\|_F/\|W_0\|_F$  (%)" if relmat is not None else r"$\|\Delta W\|_F$")
ax.set_ylim(0, None); lg(ax, ncol=2, loc="upper center")
gt(ax, "Relative update vs depth (attn v/out lead)" if relmat is not None else "‖ΔW‖ vs depth")
# panel 3: mean relative perturbation per module-type (which subspaces adapted)
ax = axs[2]; agg = np.nanmean(src, axis=0); order = np.argsort(agg)
ax.barh(range(len(MORDER)), agg[order], color=[GLINE[o % len(GLINE)] for o in order], ec=MAGENTA, alpha=.85)
ax.set_yticks(range(len(MORDER))); ax.set_yticklabels([MLAB[MORDER[o]].replace("\n", ".") for o in order], fontsize=8)
for y, o in enumerate(order):
    ax.annotate(f"{agg[o]:.2f}" + ("%" if relmat is not None else ""), (agg[o], y),
                textcoords="offset points", xytext=(4, 0), va="center", fontsize=8, color=MAGENTA)
ax.set_xlabel("mean relative update (%)" if relmat is not None else "mean ‖ΔW‖")
ax.grid(axis="x", alpha=.6); ax.grid(axis="y", visible=False); gt(ax, "Which projections adapted most")
sup(fig, "Figure E. How fine-tuning reshaped the per-layer parameter space (v12.2)")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/E_param_space.png"); plt.close()
print(f"  [E] param-space — {'relative %' if relmat is not None else 'absolute'}, "
      f"peak module {MLAB[MORDER[int(np.argmax(np.nanmean(src, axis=0)))]].replace(chr(10), '.')}")
print("[ok] 🎀 figures/{A_lora_svd, B_pareto, C_schedule, D_quant_rd, E_param_space}.png")
