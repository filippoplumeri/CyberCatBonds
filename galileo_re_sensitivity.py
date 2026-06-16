"""
Sensitivity
  (A) how much more can the footprint trigger close if the tiers are cleaner?
  (B) which parameter moves the spread the most? -> tornado
  (C) does indirect beats direct survive different covenant assumptions?

campione più leggero (1M years) because there are a lot of runs.
"""
import copy, numpy as np, pandas as pd
from scipy import stats, optimize, integrate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from galileo_re_model import P, simulate, conditional_variance, expected_loss, OUT

N = P["N_NOTIONAL"]; GRID = np.linspace(0, 1, 2001)
NY = 1_000_000                                   # lighter sample

def sim(sigma=None, df=None, freq=None, eta=None):
    # run the model
    p = copy.deepcopy(P); p["N_YEARS"] = NY
    if sigma is not None: p["LN_SIGMA"]  = sigma
    if df    is not None: p["COPULA_DF"] = df
    if freq  is not None: p["LAMBDA"]    = freq
    if eta   is not None: p["ETA_SIGMA"] = eta
    return simulate(p, copula="t", seed=1)

def surv(L):
    L = np.sort(L); return 1.0 - np.searchsorted(L, GRID, side="right")/len(L)
def wEL(S, lam):
    Sc = np.clip(S, 1e-9, 1-1e-9); return integrate.trapezoid(stats.norm.cdf(stats.norm.ppf(Sc)+lam), GRID)
def spread(lam, sigma=None, df=None, freq=None):
    return wEL(surv(sim(sigma, df, freq)["yr_payout"]/N), lam)

# same calibration
lam = optimize.brentq(lambda l: spread(l, sigma=1.15) - 0.0925, 0, 10)
AMB = np.round(np.arange(1.00, 1.29, 0.04), 2)
def robust(lam=lam, df=None, freq=None, amb_hi=1.28):
    # worst case over the ambiguity set, up to amb_hi
    grid = np.round(np.arange(1.00, amb_hi+0.001, 0.04), 2)
    return max(spread(lam, sigma=s, df=df, freq=freq) for s in grid)

base_robust = robust()
print(f"calibrated lambda = {lam:.3f},  base robust spread = {100*base_robust:.2f}%")

# --- (A) footprint value vs how clean the tiers are ------------------
# smaller within-tier noise -> the footprint tells you more -> closes more
etas = [0.30, 0.40, 0.45, 0.55, 0.70]
pct_closed = []
for e in etas:
    s = sim(eta=e)
    cv_p = conditional_variance(s["i_param"], s["ls"])
    cv_h = conditional_variance(s["theta"],   s["ls"])
    pct_closed.append(100*(cv_p - cv_h)/cv_p)
print("\n(A) basis-risk reduction vs eta_sigma:")
for e, p_ in zip(etas, pct_closed): print(f"   eta={e}: {p_:.1f}% of the gap closed")

fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(etas, pct_closed, "o-", color="#1f5c8b", lw=1.9)
ax.axvline(0.45, ls="--", lw=.8, color="grey")
ax.annotate("base case\n(16%)", xy=(0.45, pct_closed[2]), xytext=(0.52, pct_closed[2]+4),
            fontsize=8.5, color="grey")
ax.scatter([0.30], [pct_closed[0]], color="#c0392b", zorder=5, s=70,
           label="favourable configuration")
ax.set_xlabel("Within-tier residual dispersion $\\sigma_\\eta$")
ax.set_ylabel("Basis-risk gap closed by footprint (%)")
ax.set_title("The footprint modifier's value grows as tiers become more homogeneous")
ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.25)
ax.invert_xaxis()                                 # cleaner tiers on the right reads better
fig.tight_layout(); fig.savefig(f"{OUT}/fig_basis_sensitivity.png", dpi=150)

# --- (B) tornado: what moves the spread ------------------------------
# low/high , copula df range,
# bands around the base value
perturb = {
    "Market price of risk $\\lambda$":      ("lam",   lam-0.10, lam+0.10),
    "Ambiguity set upper bound":            ("amb",   1.24,     1.32),
    "Copula d.o.f. $\\nu$":                 ("df",    5.0,      2.0),
    "Event frequency $\\lambda_{freq}$":    ("freq",  1.0,      1.4),
    "Severity centre $\\sigma_{\\ln}$":     ("sig",   1.10,     1.20),
}
rows = []
for name, (kind, lo, hi) in perturb.items():
    if kind == "lam":
        s_lo, s_hi = robust(lam=lo), robust(lam=hi)
    elif kind == "amb":
        s_lo, s_hi = robust(amb_hi=lo), robust(amb_hi=hi)
    elif kind == "df":
        s_lo, s_hi = robust(df=lo), robust(df=hi)
    elif kind == "freq":
        s_lo, s_hi = robust(freq=lo), robust(freq=hi)
    elif kind == "sig":
    
        s_lo = max(spread(lam, sigma=x) for x in np.round(np.arange(0.95,lo+0.14,0.04),2))
        s_hi = max(spread(lam, sigma=x) for x in np.round(np.arange(1.05,hi+0.14,0.04),2))
    rows.append((name, 100*s_lo, 100*s_hi))

rows.sort(key=lambda r: abs(r[2]-r[1]))           # biggest swing on top
fig, ax = plt.subplots(figsize=(7.4, 4.6))
base = 100*base_robust
for i,(name,lo,hi) in enumerate(rows):
    a,b = min(lo,hi), max(lo,hi)
    ax.barh(i, b-a, left=a, color="#1f5c8b", alpha=.85, height=.6)
    ax.text(a+0.08, i, f"{a:.1f}", va="center", ha="left",  fontsize=8, color="white")
    ax.text(b+0.10, i, f"{b:.1f}", va="center", ha="left",  fontsize=8, color="black")
ax.axvline(base, color="#c0392b", lw=1.3, ls="--", label=f"base case ({base:.1f}%)")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(11.0, 16.2); ax.set_xlabel("Robust fair spread (%)")
ax.set_title("Sensitivity of the robust fair spread")
ax.legend(frameon=False, fontsize=9, loc="lower right"); ax.grid(alpha=.25, axis="x")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_tornado.png", dpi=150)
print("\n(B) tornado rows (name, low, high):")
for r in rows: print(f"   {r[0][:30]:32s} {r[1]:6.2f}  {r[2]:6.2f}")

# --- (C) does indirect stay ahead of direct? -------------------------
# redo the covenant split
print("\n(C) covenant: direct vs indirect under different assumptions")
amb_base = base_robust - spread(lam, sigma=1.15)
for fred, contract in [(0.20,0.40),(0.10,0.40),(0.30,0.40),(0.20,0.25),(0.20,0.55)]:
    LS = (1-fred)*P["LAMBDA"]
    s_str = spread(lam, sigma=1.15, freq=LS)
    direct = spread(lam, sigma=1.15) - s_str
    mid, half = 1.14, 0.14
    hi = round(mid + (1-contract)*half, 2)
    amb_mon = spread(lam, sigma=hi, freq=LS) - s_str
    indirect = amb_base - amb_mon
    print(f"   freq -{int(fred*100)}%, set -{int(contract*100)}%: direct={100*direct:.2f}%  indirect={100*indirect:.2f}%")

pd.Series({"lambda":lam,"base_robust_pct":100*base_robust,
           "pct_closed_eta030":pct_closed[0],"pct_closed_eta045":pct_closed[2],
           "pct_closed_eta070":pct_closed[-1]}).round(3).to_csv(f"{OUT}/galileo_re_sensitivity.csv", header=["value"])
print("\nsaved figures + table")
