"""Diverse gyaru figures from accumulated logs (SFT val curves, ORPO loss, eval metrics).
🎀 style matched to fig1_val_curves (Apple SD Gothic Neo, no bold, pink/yellow). -> figures/*.png"""
import sys, re, json, glob, os
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PINK = "#ee1d8f"; MAGENTA = "#b3146e"; HOTPINK = "#ff3d9a"; LIGHTPINK = "#f3c9df"
YELLOW = "#ffd93d"; ORANGE = "#ffbf3d"; DARKMAG = "#a8136a"; CHARCOAL = "#5d3a4a"; GREEN = "#2ca089"
GCMAP = LinearSegmentedColormap.from_list("gy", ["#ffe14d", "#ffbf3d", "#ff9ec9", "#ff5fa6", "#ff3d9a", "#ee1d8f", "#a8136a"])
# DISTINCT warm palette for many-line plots (smooth gradient blurs adjacent lines). + per-line marker/linestyle.
GLINE = ["#ffce1f", "#ff9a00", "#ff6a2f", "#ff5277", "#ff2d95", "#e0218a", "#b3146e", "#8e1b66", "#ffb347", "#ff86b0", "#c9559b"]
GLS = ["-", "--", "-."]; GMK = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<"]
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 11, "font.family": ["Apple SD Gothic Neo", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False, "axes.grid": True, "grid.color": LIGHTPINK, "grid.alpha": .6, "grid.linewidth": .7,
    "axes.edgecolor": "#d98cb5", "axes.linewidth": 1.2, "axes.labelcolor": CHARCOAL, "axes.titlecolor": MAGENTA,
    "axes.titleweight": "normal", "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "text.color": CHARCOAL, "xtick.color": CHARCOAL, "ytick.color": CHARCOAL,
})
def gt(ax, t, fs=11): ax.set_title(t, fontsize=fs, color=MAGENTA)
def sup(fig, t): fig.suptitle(t, fontsize=13, color=MAGENTA)
def lg(ax, **k):
    l = ax.legend(fontsize=8, framealpha=.95, **k); l.get_frame().set_edgecolor(LIGHTPINK); return l
def smooth(y, w=5):
    y = np.asarray(y, float)
    if len(y) < w: return y
    pad = w // 2  # edge-pad (not zero) so boundaries don't artificially dip toward 0
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")
def val_curve(path):
    t = open(path).read()
    return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"Iter (\d+): Val loss ([\d.]+)", t)]
def orpo_curve(path):
    t = open(path).read()
    return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"iter (\d+)/\d+\s+loss ([\d.]+)", t)]

det = json.load(open("logs/eval/det_metrics.json")); sem = json.load(open("logs/eval/sem_metrics.json"))
div = json.load(open("logs/eval/diversity_metrics.json")); cap = json.load(open("logs/eval/cap_metrics.json"))

# ============ fig1_val_curves (REDRAW, v2->v12.2) ============
SFT = ["v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10", "v11", "v12", "v122"]
LBL = {"v122": "v12.2", "v121": "v12.1"}  # file key -> proper version label
def vl_label(v): return LBL.get(v, v)
cols = {v: GLINE[i % len(GLINE)] for i, v in enumerate(SFT)}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [2, 1]})
bests = []
for i, v in enumerate(SFT):
    c = val_curve(f"logs/train/sft_{v}.log")
    if len(c) < 3: continue
    it, vl = zip(*c); col = cols[v]; ls = GLS[i % len(GLS)]
    axL.plot(it, vl, color=col, lw=.4, alpha=.18)
    axL.plot(it, smooth(np.array(vl)), color=col, lw=1.4, ls=ls, label=vl_label(v))
    bi, bv = min(c, key=lambda x: x[1]); axL.scatter([bi], [bv], color=col, marker="*", s=240, zorder=6, ec="white", lw=1.2)
    bests.append((v, bv))
axL.set_xlabel("training iteration"); axL.set_ylabel("validation loss"); axL.set_ylim(1.1, 3.1)
axL.text(.01, .015, "thin=raw, thick=smoothed(w=5), ★=val-min checkpoint", transform=axL.transAxes, fontsize=7.5, color=CHARCOAL, style="italic")
lg(axL, ncol=3, loc="upper right", title="version")
gt(axL, "Validation-loss convergence across iterations")
vs = [b[0] for b in bests]; bvv = [b[1] for b in bests]; vsd = [vl_label(v) for v in vs]
axR.bar(vsd, bvv, color=[cols[v] for v in vs], ec="white")
for i, val in enumerate(bvv): axR.annotate(f"{val:.3f}", (i, val), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=MAGENTA)
axR.set_ylim(0, 2.95); axR.set_ylabel("best (val-min) loss"); axR.tick_params(axis="x", rotation=0, labelsize=8)
gt(axR, "Best checkpoint per stage")
sup(fig, "Figure 2. SFT validation convergence (v2→v12.2)")
plt.tight_layout(rect=[0, 0, 1, .95])
if os.path.exists("figures/02_val_curves.png"): os.remove("figures/fig1_val_curves.png")
plt.savefig("figures/02_val_curves.png"); plt.close()

# ============ fig_orpo_curves: ORPO loss curves + final per stage ============
ORPO = [("orpo2", "v8"), ("orpo_v9", "v9"), ("orpo_v10", "v10"), ("orpo_v11", "v11"), ("orpo_v12", "v12"), ("orpo_v121", "v12.1")]
# v122 orpo iters live in the pipeline log
v122c = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"iter (\d+)/\d+\s+loss ([\d.]+)", open("logs/train/v122_pipeline.log").read())]
ocols = {lab: GLINE[i % len(GLINE)] for i, (_, lab) in enumerate(ORPO + [("", "v12.2")])}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.8, 1]})
series = []
for i, (f, lab) in enumerate(ORPO):
    c = orpo_curve(f"logs/train/{f}.log")
    if not c: continue
    it, lo = zip(*c); axL.plot(it, smooth(np.array(lo), 3), color=ocols[lab], lw=1.5, ls=GLS[i % len(GLS)], marker=GMK[i % len(GMK)], ms=4, label=lab); series.append((lab, lo[-1]))
if v122c:
    it, lo = zip(*v122c); axL.plot(it, smooth(np.array(lo), 3), color=PINK, lw=2.0, marker="*", ms=8, label="v12.2 ★"); series.append(("v12.2", lo[-1]))
axL.set_xlabel("ORPO iteration"); axL.set_ylabel("ORPO loss (noisy, per-pair)"); lg(axL, ncol=2, loc="upper right")
gt(axL, "Custom MLX ORPO loss curves (λ=0.3)")
labs = [s[0] for s in series]; fin = [s[1] for s in series]
axR.bar(labs, fin, color=[ocols.get(l, PINK) for l in labs], ec="white")
axR.set_ylabel("final-iter loss"); axR.tick_params(axis="x", rotation=0, labelsize=8)
gt(axR, "Final ORPO loss per stage")
sup(fig, "Figure 11. Preference-optimization (ORPO) training dynamics")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/11_orpocurves.png"); plt.close()

# ============ fig_tradeoff: surface↔substance & diversity↔style ============
EV = [("v6", "v6"), ("orpo2", "v8"), ("v9orpo", "v9"), ("v10orpo", "v10"), ("v11orpo", "v11"), ("v12orpo", "v12"), ("v121orpo", "v12.1"), ("v122orpo", "v12.2")]
ecol = {t: GLINE[i % len(GLINE)] for i, (t, _) in enumerate(EV)}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
for t, lab in EV:
    x, y = det[t]["marker_dens"], sem[t]["acc_style"]
    axL.scatter(x, y, s=190, color=ecol[t], ec="white", lw=1.6, zorder=5)
    axL.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8.5, color=ecol[t])
axL.set_xlabel("surface marker density / 100c"); axL.set_ylabel("Acc_style (substance)")
axL.text(.98, .03, "more markers ≠ more gyaru-ness\n(surface vs substance)", transform=axL.transAxes, ha="right", fontsize=8, color=CHARCOAL, style="italic")
gt(axL, "Surface ↔ substance")
for t, lab in EV:
    x, y = div[t]["d2"], sem[t]["acc_style"]
    axR.scatter(x, y, s=190, color=ecol[t], ec="white", lw=1.6, zorder=5)
    axR.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8.5, color=ecol[t])
axR.scatter(div["v122orpo"]["d2"], sem["v122orpo"]["acc_style"], s=420, marker="*", color=PINK, ec="white", lw=1.5, zorder=6)
axR.set_xlabel("distinct-2 (diversity)"); axR.set_ylabel("Acc_style (style)")
gt(axR, "Diversity ↔ style (Pareto)")
sup(fig, "Figure 7. Capability trade-offs across the lineage")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/07_tradeoff.png"); plt.close()

# ============ fig_J_safety: J decomposition & crisis safety ============
xs = list(range(len(EV))); labs = [l for _, l in EV]
acc = [sem[t]["acc_style"] for t, _ in EV]; sim = [sem[t]["sim"] for t, _ in EV]; J = [sem[t]["J"] for t, _ in EV]
cdhr = [det[t]["cdhr"] for t, _ in EV]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
axL.plot(xs, acc, "-o", color=HOTPINK, lw=2.2, ms=6, label="Acc_style")
axL.plot(xs, sim, "-s", color=ORANGE, lw=2.2, ms=6, label="Sim")
axL.plot(xs, J, "-D", color=MAGENTA, lw=2.8, ms=7, label="J = Acc·Sim")
axL.scatter(7, J[-1], s=300, marker="*", color=PINK, ec="white", lw=1.3, zorder=6)
axL.set_xticks(xs); axL.set_xticklabels(labs, fontsize=8); axL.set_ylim(.4, 1); axL.set_ylabel("score"); lg(axL, loc="center right")
gt(axL, "J = Acc_style × Sim decomposition")
bars = axR.bar(xs, cdhr, color=[GREEN if c <= .34 else ORANGE if c <= .5 else "#e07b9a" for c in cdhr], ec="white")
axR.axhline(.34, ls="--", color=GREEN, lw=1.2); axR.annotate("safe zone", (0, .30), fontsize=8, color=GREEN)
axR.set_xticks(xs); axR.set_xticklabels(labs, fontsize=8); axR.set_ylabel("CDHR (crisis-deflection harm, raw)"); axR.set_ylim(0, 1)
gt(axR, "Crisis-safety evolution")
sup(fig, "Figure 5. Style–meaning composite & safety")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/05_jsafety.png"); plt.close()

# ============ fig_capability: COPA & ppl (forgetting) ============
copa = [cap[t]["copa_acc"] for t, _ in EV]; ppl = [cap[t]["neutral_ppl"] for t, _ in EV]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.4))
axL.bar(xs, copa, color=[ecol[t] for t, _ in EV], ec="white")
axL.axhline(.74, ls="--", color=DARKMAG, lw=1.4); axL.annotate("base 7.8B ≈ .74", (0, .755), fontsize=8, color=DARKMAG)
for i, c in enumerate(copa): axL.annotate(f"{c:.2f}", (i, c), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=CHARCOAL)
axL.set_xticks(xs); axL.set_xticklabels(labs, fontsize=8); axL.set_ylim(.6, .8); axL.set_ylabel("KoBEST-COPA accuracy")
gt(axL, "Commonsense retention (no essential forgetting)")
axR.plot(xs, ppl, "-o", color=PINK, lw=2.6, ms=7, mec="white")
axR.scatter(7, ppl[-1], s=300, marker="*", color=PINK, ec="white", lw=1.3, zorder=6)
axR.fill_between(xs, ppl, 38, color=YELLOW, alpha=.18)
axR.axhline(42, ls="--", color=DARKMAG, lw=1.2); axR.annotate("gate ≤42", (5.5, 43), fontsize=8, color=DARKMAG)
axR.set_xticks(xs); axR.set_xticklabels(labs, fontsize=8); axR.set_ylabel("neutral perplexity (↓ better)")
gt(axR, "Neutral fluency (style–fluency trade-off)")
sup(fig, "Figure 6. Capability preservation (COPA & perplexity)")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/06_capability.png"); plt.close()

# ============ fig_firing: trigger P/R & surface composition ============
yR = [det[t]["yaho_R"] for t, _ in EV]; pR = [det[t]["para_R"] for t, _ in EV]; fF = [det[t]["false_fire"] for t, _ in EV]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.4))
w = .27; xa = np.arange(len(EV))
axL.bar(xa - w, yR, w, color=YELLOW, ec="white", label="yaho recall")
axL.bar(xa, pR, w, color=HOTPINK, ec="white", label="parapara recall")
axL.bar(xa + w, fF, w, color=DARKMAG, ec="white", label="false-fire ↓")
axL.set_xticks(xa); axL.set_xticklabels(labs, fontsize=8); axL.set_ylim(0, 1.05); axL.set_ylabel("rate"); lg(axL, ncol=3, loc="upper center")
gt(axL, "Trigger recall & false-fire")
mk = [det[t]["marker_dens"] for t, _ in EV]; em = [det[t]["emoji_dens"] for t, _ in EV]; ti = [det[t]["tilde_dens"] for t, _ in EV]
axR.bar(xa, mk, color=PINK, ec="white", label="markers")
axR.bar(xa, em, bottom=mk, color=ORANGE, ec="white", label="emoji")
axR.bar(xa, ti, bottom=np.array(mk) + np.array(em), color=YELLOW, ec="white", label="tilde ~")
axR.set_xticks(xa); axR.set_xticklabels(labs, fontsize=8); axR.set_ylabel("density / 100 chars (stacked)"); lg(axR, loc="upper right")
gt(axR, "Surface-marker composition")
sup(fig, "Figure 3. Trigger firing & surface composition")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/03_firing.png"); plt.close()

print("[ok] 🎀 figures/{02_val_curves,11_orpocurves,07_tradeoff,05_jsafety,06_capability,03_firing}.png".replace("🎀 ", ""))
