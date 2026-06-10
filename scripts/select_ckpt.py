"""Auto-select the SFT val-min checkpoint (3-point smoothed, robust to noise; prefer >=1 epoch
for long-context absorption) and promote it to <adapter_dir>/adapters.safetensors + SELECTED.txt.
Usage: python scripts/select_ckpt.py <train_log> <adapter_dir> <epoch_iters>"""
import sys, re, os, glob, shutil

log, adir = sys.argv[1], sys.argv[2]
epoch_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 0
txt = open(log).read()
vals = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"Iter (\d+): Val loss ([\d.]+)", txt)]
have = {int(os.path.basename(f).split("_")[0]) for f in glob.glob(f"{adir}/*_adapters.safetensors")}
# 3-point smoothed val
sm = []
for i, (it, v) in enumerate(vals):
    lo, hi = max(0, i - 1), min(len(vals), i + 2)
    sm.append((it, sum(x for _, x in vals[lo:hi]) / (hi - lo)))
# prefer checkpoints that exist on disk AND >= 1 epoch (so long-context seen); fall back to any
cand = [(it, s) for it, s in sm if it in have and (epoch_iters == 0 or it >= epoch_iters)]
if not cand:
    cand = [(it, s) for it, s in sm if it in have]
best_it = min(cand, key=lambda x: x[1])[0]
raw = dict(vals)[best_it]
src = f"{adir}/{best_it:07d}_adapters.safetensors"
shutil.copy(src, f"{adir}/adapters.safetensors")
open(f"{adir}/SELECTED.txt", "w").write(
    f"AUTO-selected it{best_it} (raw val {raw:.3f}, 3pt-smoothed-min, >=1ep preferred). {adir}\n")
print(f"[select] {adir}: it{best_it} (raw {raw:.3f}) -> adapters.safetensors")
# trim numbered ckpts to reclaim disk (val curve is in the log)
for f in glob.glob(f"{adir}/*_adapters.safetensors"):
    os.remove(f)
print(f"[select] trimmed {len(have)} numbered ckpts")
