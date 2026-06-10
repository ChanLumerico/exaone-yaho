"""README tech-report figures from REAL eval data + corpus stats. -> figures/rf_*.png
GYARU STYLE (pink+yellow, magenta titles, light-pink grid, ★ markers) +
multi-panel (>=2 horizontal subplots), information-dense. English/romaji labels."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette + font matched to figures/fig1_val_curves.png (make_figures.py). NO bold (per request).
PINK = "#ee1d8f"; MAGENTA = "#b3146e"; HOTPINK = "#ff3d9a"; LIGHTPINK = "#f3c9df"
YELLOW = "#ffd93d"; ORANGE = "#ffbf3d"; DARKMAG = "#a8136a"; CHARCOAL = "#5d3a4a"
GYARU = ["#ffd93d", "#ffbf3d", "#ff9ec9", "#ff5fa6", "#ff3d9a", "#ee1d8f", "#c9187a", "#a8136a"]
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 11, "font.family": ["Apple SD Gothic Neo", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False, "axes.grid": True,
    "grid.color": LIGHTPINK, "grid.alpha": .6, "grid.linewidth": .7,
    "axes.edgecolor": "#d98cb5", "axes.linewidth": 1.2, "axes.labelcolor": CHARCOAL,
    "axes.titlecolor": MAGENTA, "axes.titleweight": "normal",
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "text.color": CHARCOAL, "xtick.color": CHARCOAL, "ytick.color": CHARCOAL,
    "axes.prop_cycle": plt.cycler(color=GYARU),
})
def gt(ax, t, fs=10.5): ax.set_title(t, fontsize=fs, color=MAGENTA)
def suptitle(fig, t): fig.suptitle(t, fontsize=12.5, color=MAGENTA)
def leg(ax, **k):
    l = ax.legend(fontsize=8, framealpha=.95, **k); l.get_frame().set_edgecolor(LIGHTPINK); return l

det = json.load(open("logs/eval/det_metrics.json")); sem = json.load(open("logs/eval/sem_metrics.json"))
div = json.load(open("logs/eval/diversity_metrics.json")); cap = json.load(open("logs/eval/cap_metrics.json"))
hpo = json.load(open("logs/eval/hpo_results.json"))

# ============ Fig 1: lineage evolution (lines) + v12.2 final bars ============
order = [("v6", "v6"), ("orpo2", "v8"), ("v9orpo", "v9"), ("v10orpo", "v10"),
         ("v11orpo", "v11"), ("v12orpo", "v12"), ("v121orpo", "v12.1"), ("v122orpo", "v12.2")]
xs = list(range(len(order))); labs = [l for _, l in order]
firing = [det[t]["macro_firing_F1"] for t, _ in order]; ff = [det[t]["false_fire"] for t, _ in order]
acc = [sem[t]["acc_style"] for t, _ in order]; ppl = [cap[t]["neutral_ppl"] for t, _ in order]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.7, 1]})
axL.plot(xs, firing, "-o", color=HOTPINK, lw=2.6, ms=6, label="firing F1 ↑")
axL.plot(xs, acc, "-s", color=ORANGE, lw=2.6, ms=6, label="Acc_style ↑")
axL.plot(xs, ff, "-^", color=DARKMAG, lw=2.6, ms=6, label="false-fire ↓")
axL.plot(7, firing[-1], "*", color=PINK, ms=22, mec="white", mew=1, zorder=5)
axL.set_ylim(0, 1); axL.set_ylabel("F1 / Acc / false-fire"); axL.set_xticks(xs); axL.set_xticklabels(labs, fontsize=8)
ax2 = axL.twinx(); ax2.plot(xs, ppl, "--D", color=YELLOW, lw=2, ms=6, mec=ORANGE, label="neutral ppl ↓")
ax2.set_ylabel("perplexity", color=ORANGE); ax2.grid(False); ax2.tick_params(colors=ORANGE)
axL.axvspan(6.5, 7.5, color=LIGHTPINK, alpha=.25)
h1, l1 = axL.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
leg(axL, loc="center left", handles=h1 + h2, labels=l1 + l2)
gt(axL, "Behavioral-metric evolution (v6→v12.2)")
# right: v12.2 final gate bars
gm = [("recall", 1.0), ("firing", firing[-1]), ("1-false_fire", 1 - ff[-1]), ("Acc_style", acc[-1]),
      ("distinct-2", div["v122orpo"]["d2"]), ("COPA", cap["v122orpo"]["copa_acc"])]
gy = np.arange(len(gm)); cols = [GYARU[i] for i in range(len(gm))]
axR.barh(gy, [v for _, v in gm], color=cols, ec="white", lw=.8)
for i, (n, v) in enumerate(gm): axR.text(v + .02, i, f"{v:.2f}", va="center", fontsize=8, color=CHARCOAL)
axR.set_yticks(gy); axR.set_yticklabels([n for n, _ in gm], fontsize=8); axR.set_xlim(0, 1.15); axR.invert_yaxis()
gt(axR, "v12.2 (FINAL ★) scores")
suptitle(fig, "Figure 1. Alignment-lineage convergence & final capabilities")
plt.tight_layout(rect=[0, 0, 1, .95]); plt.savefig("figures/01_lineage.png"); plt.close()

# ============ Fig 2: radar + gate-vs-threshold bars ============
axes = ["recall", "firing", "1-false_fire", "Acc_style", "distinct-2", "COPA", "fluency"]
def npp(p): return max(0, min(1, (60 - p) / (60 - 25)))
def radar(t, r): return [r, det[t]["macro_firing_F1"], 1 - det[t]["false_fire"], sem[t]["acc_style"], div[t]["d2"], cap[t]["copa_acc"], npp(cap[t]["neutral_ppl"])]
v122, v121 = radar("v122orpo", 1.0), radar("v121orpo", 0.33)
fig = plt.figure(figsize=(12, 4.6))
axP = fig.add_subplot(1, 2, 1, polar=True); axP.set_facecolor("#fff6fa")
ang = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist(); ang += ang[:1]
for vals, c, lb in [(v121, ORANGE, "v12.1"), (v122, PINK, "v12.2 ★")]:
    vv = vals + vals[:1]; axP.plot(ang, vv, "-o", color=c, lw=2.4, ms=5, label=lb); axP.fill(ang, vv, color=c, alpha=.18)
axP.set_xticks(ang[:-1]); axP.set_xticklabels(axes, fontsize=8); axP.set_ylim(0, 1)
axP.grid(color=LIGHTPINK, alpha=.5); axP.spines["polar"].set_color(LIGHTPINK)
axP.set_title("normalized capability radar", fontsize=10.5, color=MAGENTA, pad=18)
leg(axP, loc="upper right", bbox_to_anchor=(1.32, 1.13))
# right: v12.2 metric vs acceptance gate
axB = fig.add_subplot(1, 2, 2)
gates = [("firing≥.80", det["v122orpo"]["macro_firing_F1"], .80, True), ("false-fire≤.10", det["v122orpo"]["false_fire"], .10, False),
         ("ppl≤42", cap["v122orpo"]["neutral_ppl"] / 60, 42 / 60, False), ("Acc≥.85", sem["v122orpo"]["acc_style"], .85, True),
         ("distinct≥.65", div["v122orpo"]["d2"], .65, True), ("recall≥.66", 1.0, .66, True)]
yy = np.arange(len(gates))
for i, (n, v, th, hi) in enumerate(gates):
    ok = (v >= th) if hi else (v <= th)
    axB.barh(i, v, color=(HOTPINK if ok else YELLOW), ec="white", lw=.8)
    axB.plot([th, th], [i - .4, i + .4], color=DARKMAG, lw=2)
axB.set_yticks(yy); axB.set_yticklabels([n for n, *_ in gates], fontsize=8); axB.invert_yaxis(); axB.set_xlim(0, 1)
axB.plot([], [], color=DARKMAG, lw=2, label="gate threshold"); leg(axB, loc="lower right")
gt(axB, "v12.2 vs acceptance gates")
suptitle(fig, "Figure 4. v12.2 capability radar & gate compliance")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/04_radar.png"); plt.close()

# ============ Fig 3: corpus marker rebalance + generation-level effect ============
G = {"agemizawa": ["아게미자와", "あげみざわ"], "maji": ["마지", "マジ"], "cho": ["쵸", "超"], "bari": ["바리", "ばり"],
     "yaba": ["야바", "ヤバ"], "nanka": ["뭔가", "なんか"], "egui": ["에구이", "えぐい"], "meccha": ["멧챠", "めっちゃ"],
     "gachi": ["가치", "ガチ"], "teka": ["테카", "てか"], "kya": ["꺄–", "きゃー"], "weei": ["웨이", "うぇーい"]}
def asst(f): return "".join(m["content"] for d in (json.loads(l) for l in open(f)) for m in d["messages"] if m["role"] == "assistant")
a0, a1 = asst("data/gyaru_corpus_12.jsonl"), asst("data/gyaru_corpus_12_2.jsonl")
ks = list(G); b = [sum(a0.count(x) for x in G[k]) for k in ks]; af = [sum(a1.count(x) for x in G[k]) for k in ks]
o = np.argsort(b)[::-1]; ks = [ks[i] for i in o]; b = [b[i] for i in o]; af = [af[i] for i in o]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.7, 1]})
x = np.arange(len(ks)); w = .4
axL.bar(x - w / 2, b, w, color=YELLOW, ec=ORANGE, lw=.8, label="corpus_12 (before)")
axL.bar(x + w / 2, af, w, color=HOTPINK, ec=MAGENTA, lw=.8, label="corpus_12.2 (after)")
axL.axhline(350, ls="--", color=PINK, lw=1.4); axL.annotate("CAP=350", (len(ks) - 1.7, 380), color=PINK, fontsize=8)
axL.set_xticks(x); axL.set_xticklabels(ks, rotation=35, ha="right", fontsize=8); axL.set_ylabel("count in corpus"); leg(axL)
gt(axL, "Corpus marker distribution (flattened)")
# right: generation-level effect (from test.txt before/after)
eff = [("agemizawa\n(in output)", 11, 4), ("choberi\nsurfaced", 0, 3), ("marker\ntypes", 28, 35)]
ex = np.arange(len(eff));
axR.bar(ex - w / 2, [e[1] for e in eff], w, color=YELLOW, ec=ORANGE, lw=.8, label="v12.1")
axR.bar(ex + w / 2, [e[2] for e in eff], w, color=HOTPINK, ec=MAGENTA, lw=.8, label="v12.2")
for i, e in enumerate(eff):
    axR.text(i - w / 2, e[1] + .4, e[1], ha="center", fontsize=8, color=CHARCOAL); axR.text(i + w / 2, e[2] + .4, e[2], ha="center", fontsize=8, color=MAGENTA)
axR.set_xticks(ex); axR.set_xticklabels([e[0] for e in eff], fontsize=8); axR.set_ylabel("count in 29-probe test"); leg(axR)
gt(axR, "Generation-level effect")
suptitle(fig, "Figure 8. Marker rebalancing: corpus distribution → generation diversity")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/08_marker.png"); plt.close()

# ============ Fig 4: quant size + compression ratio ============
bits = [3, 4, 5, 6, 8, 16]; size = [3.7, 4.6, 5.5, 6.4, 8.2, 15.0]; eff = [4, 5, 6, 7, 9, 16]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4))
axL.plot(bits[:-1], size[:-1], "-o", color=HOTPINK, lw=2.6, ms=10, mec="white", mew=1, label="MLX quant")
axL.plot(16, 15, "*", color=PINK, ms=24, mec="white", mew=1, label="bf16 fused ★")
for bb, ss, ee in zip(bits, size, eff): axL.annotate(f"{ss}GB\n~{ee}b", (bb, ss), fontsize=7, ha="center", va="bottom", xytext=(0, 7), textcoords="offset points")
axL.set_xlabel("nominal bits"); axL.set_ylabel("on-disk size (GB)"); axL.set_xticks(bits); leg(axL, loc="upper left")
gt(axL, "Bit-width vs size")
ratio = [15.0 / s for s in size]; bcol = [GYARU[i] for i in range(len(bits))]
axR.bar([f"q{b}" if b < 16 else "bf16" for b in bits], ratio, color=bcol, ec="white", lw=.8)
for i, r in enumerate(ratio): axR.text(i, r + .05, f"{r:.1f}×", ha="center", fontsize=8, color=CHARCOAL)
axR.set_ylabel("compression vs bf16 (×)"); axR.axhline(1, ls=":", color=LIGHTPINK)
gt(axR, "Compression ratio (all retain persona)")
suptitle(fig, "Figure 12. MLX quantization: size & compression trade-off")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/12_quant.png"); plt.close()

# ============ Fig 5: LoRA schematic + HPO rank/lr ablation ============
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
axL.axis("off"); axL.set_xlim(0, 10); axL.set_ylim(0, 4)
def box(x, y, w, h, c, t, ts=9.5):
    axL.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="white", alpha=.95, lw=1.5, zorder=2))
    axL.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=ts, color="white", zorder=3)
box(.3, 1, 2, 2, ORANGE, "W₀\nd×k\n(frozen)")
axL.text(2.6, 2, "+", fontsize=20, ha="center", color=MAGENTA)
box(3.1, 1.4, 1.7, 1.2, HOTPINK, "B\nd×r"); axL.text(5.0, 2, "×", fontsize=16, ha="center", color=MAGENTA)
box(5.2, 1.7, 2.1, .6, PINK, "A  r×k", ts=9)
axL.text(7.7, 2, "=", fontsize=18, ha="center", color=MAGENTA)
box(8.0, 1, 1.6, 2, "#e91e63", "h")
axL.text(5, .35, r"$h = W_0 x + \frac{\alpha}{r}BAx,\ r\ll\min(d,k)$", ha="center", fontsize=11, color=CHARCOAL)
gt(axL, "LoRA low-rank reparameterization (r=16)")
# right: HPO val-min by config (real)
hm = {h["name"]: h["val_min"] for h in hpo}
names = ["r8", "lr2e5", "r32", "lr1e4"]; vals = [hm.get(n, 0) for n in names]
labels = ["r8\nscale8", "r16\nlr2e-5", "r32\nscale32", "r16\nlr1e-4"]
cols = [HOTPINK, PINK, YELLOW, ORANGE]
axR.bar(range(len(names)), vals, color=cols, ec="white", lw=.8)
for i, v in enumerate(vals): axR.text(i, v + .03, f"{v:.2f}", ha="center", fontsize=8, color=CHARCOAL)
axR.axhline(1.3, ls="--", color=DARKMAG, lw=1.2); axR.annotate("good", (3.1, 1.32), fontsize=7, color=DARKMAG)
axR.set_xticks(range(len(names))); axR.set_xticklabels(labels, fontsize=7.5); axR.set_ylabel("val-min loss"); axR.set_ylim(1, 2.4)
gt(axR, "HPO: rank/lr ablation")
suptitle(fig, "Figure 9. LoRA reparameterization & hyperparameter selection")
plt.tight_layout(rect=[0, 0, 1, .93]); plt.savefig("figures/09_lora.png"); plt.close()

# ============ Fig 6: ORPO penalty + preference probability ============
d = np.linspace(-5, 6, 400); sig = 1 / (1 + np.exp(-d)); L = -np.log(sig)
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))
axL.plot(d, L, color=PINK, lw=3)
axL.fill_between(d, L, where=(d < 0), color=HOTPINK, alpha=.18); axL.fill_between(d, L, where=(d >= 0), color=YELLOW, alpha=.18)
axL.axvline(0, ls=":", color=LIGHTPINK, lw=1.5)
axL.annotate("chosen worse\n→ large penalty", (-3.4, 3.2), fontsize=8, color=MAGENTA)
axL.annotate("chosen ≫ rejected\n→ →0", (2.2, .4), fontsize=8, color=ORANGE)
axL.set_xlabel(r"$\Delta=\log\,\mathrm{odds}_\theta(y_w)-\log\,\mathrm{odds}_\theta(y_l)$"); axL.set_ylabel(r"$\mathcal{L}_{OR}=-\log\sigma(\Delta)$")
gt(axL, "Odds-ratio penalty")
axR.plot(d, sig, color=HOTPINK, lw=3, label=r"$\sigma(\Delta)$ = P(prefer $y_w$)")
axR.axhline(.5, ls=":", color=LIGHTPINK); axR.axvline(0, ls=":", color=LIGHTPINK)
axR.fill_between(d, sig, .5, where=(d >= 0), color=YELLOW, alpha=.2); axR.fill_between(d, sig, .5, where=(d < 0), color=HOTPINK, alpha=.2)
axR.scatter([0], [.5], color=PINK, s=80, marker="*", zorder=5, ec="white")
axR.set_xlabel(r"$\Delta$"); axR.set_ylabel("preference probability"); axR.set_ylim(0, 1); leg(axR, loc="lower right")
gt(axR, "Preference probability  $\\mathcal{L}_{ORPO}=\\mathcal{L}_{SFT}+\\lambda\\mathcal{L}_{OR}$")
suptitle(fig, "Figure 10. ORPO: odds-ratio penalty & induced preference (λ=0.3)")
plt.tight_layout(rect=[0, 0, 1, .94]); plt.savefig("figures/10_orpo.png"); plt.close()

print("[ok] multi-panel gyaru figures: 01_lineage,04_radar,08_marker,12_quant,09_lora,10_orpo.png")
