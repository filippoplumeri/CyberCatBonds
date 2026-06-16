"""
Pricing of Galileo Re.

  1. risk load  -> Wang transform, with lambda calibrated so the base spread sits
                   at ~9.25%
  2. ambiguity  -> I don't trust a single severity, so I price the worst case over
                   a whole set of plausible ones
  3. covenant   -> monitoring lowers the loss and shrinks the set, so it pays twice

Important: what comes out is a spread, not a dollar price.
"""
import copy, numpy as np, pandas as pd
from scipy import stats, optimize, integrate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from galileo_re_model import P, simulate, OUT

N = P["N_NOTIONAL"]
TARGET_BASE_SPREAD = 0.0925       # bottom of the observed cyber spreads
GRID = np.linspace(0, 1, 2001)    # grid for the survival function

def survival(loss_frac):
    # empirical S(x) = P(L > x) on the grid
    L = np.sort(loss_frac)
    return 1.0 - np.searchsorted(L, GRID, side="right") / len(L)

def wang_EL(S, lam):
    # push probability mass into the tail, then integrate -> risk-adjusted EL.
    # occhio: clip S away from 0 and 1
    Sc = np.clip(S, 1e-9, 1 - 1e-9)
    S_star = stats.norm.cdf(stats.norm.ppf(Sc) + lam)
    return integrate.trapezoid(S_star, GRID)

def loss_fraction(sigma_ln, lam_freq=None):
    # rerun the model with a different severity (and maybe frequency)
    p = copy.deepcopy(P)
    p["LN_SIGMA"] = sigma_ln
    if lam_freq is not None:
        p["LAMBDA"] = lam_freq
    return simulate(p, copula="t", seed=1)["yr_payout"] / N

# --- steps 1-2: calibrate, then price under ambiguity ----------------
S_base = survival(loss_fraction(P["LN_SIGMA"]))
EL_base = integrate.trapezoid(S_base, GRID)

# find the lambda that puts the base spread exactly on the target
lam = optimize.brentq(lambda l: wang_EL(S_base, l) - TARGET_BASE_SPREAD, 0.0, 10.0)
s_base = wang_EL(S_base, lam)

# the ambiguity set
sigmas = np.round(np.arange(1.00, 1.29, 0.04), 2)
spreads = {sg: wang_EL(survival(loss_fraction(sg)), lam) for sg in sigmas}
s_robust = max(spreads.values())       # worst case over the set, robust price

EL_star_base = s_base
risk_premium = EL_star_base - EL_base
ambiguity_premium = s_robust - EL_star_base

print(f"calibrated lambda          : {lam:.3f}")
print(f"physical EL                : {100*EL_base:5.2f}%")
print(f"base spread  (Wang)        : {100*s_base:5.2f}%")
print(f"robust spread (maxmin)     : {100*s_robust:5.2f}%")
print(f"  -> expected loss         : {100*EL_base:5.2f}%")
print(f"  -> risk premium          : {100*risk_premium:5.2f}%")
print(f"  -> ambiguity premium     : {100*ambiguity_premium:5.2f}%")

# --- step 3: the resilience covenant ---------------------------------
# two channels:
#   direct   = stronger posture -> fewer events -> lower EL
#   indirect = monitoring -> the ambiguity set shrinks -> smaller ambiguity premium
LAMBDA_STRONG = 0.8 * P["LAMBDA"]      # strong posture = 20% fewer events. poi lo stresso
central = P["LN_SIGMA"]                # 1.15

# direct: just reprice at the central severity with fewer events
s_strong_central = wang_EL(survival(loss_fraction(central, lam_freq=LAMBDA_STRONG)), lam)
rebate_direct = s_base - s_strong_central

# indirect: monitoring contracts the set by 40% of its width, around the centre
sigma_lo_base, sigma_hi_base = 1.00, 1.28
mid  = (sigma_lo_base + sigma_hi_base) / 2          # 1.14
half = (sigma_hi_base - sigma_lo_base) / 2          # 0.14
sigma_hi_mon = round(mid + 0.6 * half, 2)           # -> 1.22
amb_monitored = (wang_EL(survival(loss_fraction(sigma_hi_mon, lam_freq=LAMBDA_STRONG)), lam)
                 - s_strong_central)
amb_saving = ambiguity_premium - amb_monitored

total_saving = rebate_direct + amb_saving
s_robust_covenant = s_robust - total_saving

print(f"\nResilience covenant:")
print(f"  direct rebate (lower EL)         : {100*rebate_direct:5.2f}%")
print(f"  ambiguity premium, base          : {100*ambiguity_premium:5.2f}%")
print(f"  ambiguity premium, monitored     : {100*amb_monitored:5.2f}%")
print(f"  indirect saving (narrower amb.)  : {100*amb_saving:5.2f}%")
print(f"  total covenant saving            : {100*total_saving:5.2f}%")
print(f"  robust spread WITH covenant      : {100*s_robust_covenant:5.2f}%")

# --- save -------------------------------------------------------------
res = {
    "lambda": lam,
    "EL_pct": 100*EL_base,
    "base_spread_pct": 100*s_base,
    "robust_spread_pct": 100*s_robust,
    "risk_premium_pct": 100*risk_premium,
    "ambiguity_premium_pct": 100*ambiguity_premium,
    "rebate_direct_pct": 100*rebate_direct,
    "ambiguity_saving_pct": 100*amb_saving,
    "total_covenant_saving_pct": 100*total_saving,
    "robust_spread_covenant_pct": 100*s_robust_covenant,
}
pd.Series(res).round(3).to_csv(f"{OUT}/galileo_re_pricing.csv", header=["value"])

# --- figure 1: where the spread comes from ---------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.2))
comp = [EL_base, risk_premium, ambiguity_premium]
labs = ["Expected loss", "Risk premium", "Ambiguity premium"]
cols = ["#9bbcd6", "#1f5c8b", "#163d5c"]
bottom = 0
for v, l, c in zip(comp, labs, cols):
    ax.bar("Galileo Re\nfair spread", 100*v, bottom=100*bottom, color=c, label=l)
    ax.text(0, 100*(bottom + v/2), f"{100*v:.2f}%", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
    bottom += v
ax.axhspan(9.25, 13.0, color="grey", alpha=.15)    # where the real bonds priced
ax.text(0.55, 11.0, "observed cyber\ncat bond range\n(9.25%-13%)",
        fontsize=8.5, color="dimgrey")
ax.set_ylabel("Spread over risk-free (%)")
ax.set_title("Decomposition of the Galileo Re fair spread")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_xlim(-0.6, 0.9); ax.set_ylim(0, 14); ax.grid(alpha=.25, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_spread_decomposition.png", dpi=150)

# --- figure 2: spread across the ambiguity set -----------------------
sig_worst = max(spreads, key=spreads.get)
fig, ax = plt.subplots(figsize=(6.8, 4.4))
xs = list(spreads.keys()); ys = [100*spreads[s] for s in xs]
ax.plot(xs, ys, "o-", color="#1f5c8b", lw=1.8)
ax.axvline(1.15, ls="--", lw=.8, color="grey")
ax.text(1.153, min(ys)+0.4, "central\nestimate", fontsize=8, color="grey")
ax.scatter([sig_worst], [100*s_robust], color="#c0392b", zorder=5, s=70,
           label="worst-case (robust price)")
ax.set_xlabel("Severity shape $\\sigma_{\\ln}$ (ambiguity set)")
ax.set_ylabel("Fair spread (%)")
ax.set_title("Spread across the ambiguity set: the robust price is the worst case")
ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ambiguity.png", dpi=150)

# --- figure 3: the covenant, step by step ----------------------------
fig, ax = plt.subplots(figsize=(7.2, 5))
steps  = ["Robust spread\n(no covenant)", "Direct rebate\n(lower EL)",
          "Indirect saving\n(narrower\nambiguity)", "Spread with\ncovenant"]
start  = 100*s_robust
lvl1   = start - 100*rebate_direct
lvl2   = lvl1  - 100*amb_saving
ax.bar(0, start, color="#163d5c", width=.6)
ax.bar(1, 100*rebate_direct, bottom=lvl1, color="#9bbcd6", width=.6)
ax.bar(2, 100*amb_saving,    bottom=lvl2, color="#1f5c8b", width=.6)
ax.bar(3, lvl2, color="#2e8b57", width=.6)
ax.plot([0.3,0.7],[start,start], color="grey", lw=.8, ls="--")
ax.plot([1.3,1.7],[lvl1,lvl1],   color="grey", lw=.8, ls="--")
ax.plot([2.3,2.7],[lvl2,lvl2],   color="grey", lw=.8, ls="--")
ax.text(0, start+0.2, f"{start:.2f}%", ha="center", fontsize=10, fontweight="bold")
ax.text(1, lvl1+100*rebate_direct/2, f"-{100*rebate_direct:.2f}", ha="center",
        va="center", color="#163d5c", fontsize=10, fontweight="bold")
ax.text(2, lvl2+100*amb_saving/2, f"-{100*amb_saving:.2f}", ha="center",
        va="center", color="white", fontsize=10, fontweight="bold")
ax.text(3, lvl2+0.2, f"{lvl2:.2f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(steps, fontsize=9)
ax.set_ylabel("Robust fair spread (%)")
ax.set_title("How the resilience covenant lowers the spread")
ax.set_ylim(0, 14.5); ax.grid(alpha=.25, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_covenant_waterfall.png", dpi=150)
print("\nsaved tables + figures")
