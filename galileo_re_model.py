"""
Galileo Re - loss model.

The yearly loss is built from three pieces, then I just count how often, and
how hard, the bond gets hit:
  - how many big outages in a year   -> Poisson
  - how long each region stays down  -> lognormal durations, tied together with a t-copula
  - turn those downtimes into money  -> the bond payout vs the sponsor's real loss

Seed is fixed, so the numbers come out the same every run. Running the file
rebuilds the base-case tables and the figures I used in the thesis.
"""

import os, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)


# manopole
P = dict(
    # bond structure
    N_NOTIONAL = 100.0,                     # notional, USD m
    T_TIERS    = 3,                         # exposure tiers
    W          = np.array([0.50, 0.30, 0.20]),   # tier weights (exposure shares)
    E          = np.array([120.0, 70.0, 40.0]),  # tier exposures, USD m

    # trigger / payout (the index Theta lives in [0,1])
    OMEGA   = 12.0,                         # waiting period, hours
    D_MAX   = 96.0,                         # duration for a full tier hit, h
    ATTACH  = 0.37,                         # attachment on the index
    EXHAUST = 0.85,                         # exhaustion on the index

    # frequency
    LAMBDA  = 1.2,                          # systemic events per year

    # outage duration (lognormal marginal)
    LN_MEDIAN = 5.0,                        # median duration, hours
    LN_SIGMA  = 1.15,                       # shape, the heavy right tail

    # dependence across tiers (t-copula)
    RHO       = 0.60,                       # pairwise correlation
    COPULA_DF = 3.0,                        # low d.o.f. -> strong tail dependence

    # leftover spread of losses inside a tier
    ETA_SIGMA = 0.45,

    # clustering option (mixed-Poisson): gamma spread of the yearly rate.
    # only kicks in with simulate(clustered=True); smaller k = more
    CLUSTER_K = 2.0,

    N_YEARS = 2_000_000,
)


# --- building blocks -------------------------------------------------

def severity(D, omega, dmax):
    # h(D): below the waiting period nothing happens, then linear up to a full
    # hit at dmax, capped at 1
    return np.clip((D - omega) / (dmax - omega), 0.0, 1.0)


def t_copula(n, T, rho, df, rng):
    # correlated uniforms with a fat tail. the distrib mixing is exactly what
    # makes the regions fail together in the extreme
    R = np.full((T, T), rho)
    np.fill_diagonal(R, 1.0)
    Z = rng.standard_normal((n, T)) @ np.linalg.cholesky(R).T
    g = rng.chisquare(df, size=(n, 1)) / df
    return stats.t.cdf(Z / np.sqrt(g), df)


def gaussian_copula(n, T, rho, rng):
    # same correlation but no tail dependence
    R = np.full((T, T), rho)
    np.fill_diagonal(R, 1.0)
    Z = rng.standard_normal((n, T)) @ np.linalg.cholesky(R).T
    return stats.norm.cdf(Z)


def durations(U, p):
    # uniforms -> hours, through the lognormal
    return stats.lognorm.ppf(U, s=p["LN_SIGMA"], scale=p["LN_MEDIAN"])


# --- the simulation ---------------------------------------------------

def simulate(p, copula="t", clustered=False, seed=1):
    rng = np.random.default_rng(seed)
    T, W, E = p["T_TIERS"], p["W"], p["E"]
    a, e = p["ATTACH"], p["EXHAUST"]

    # how many events per year. with clustered=True the yearly rate itself is
    # random (gamma), so bad years group up; off by default
    if clustered:
        k = p["CLUSTER_K"]
        rate   = p["LAMBDA"] * rng.gamma(shape=k, scale=1.0 / k, size=p["N_YEARS"])
        counts = rng.poisson(rate)
    else:
        counts = rng.poisson(p["LAMBDA"], size=p["N_YEARS"])
    # flatten to one long list of events, remembering which year each came from
    n_events = int(counts.sum())
    year_id  = np.repeat(np.arange(p["N_YEARS"]), counts)

    # durations across the tiers, then the per-tier severity
    U = t_copula(n_events, T, p["RHO"], p["COPULA_DF"], rng) if copula == "t" \
        else gaussian_copula(n_events, T, p["RHO"], rng)
    h = severity(durations(U, p), p["OMEGA"], p["D_MAX"])

    # theta uses the real exposure weights; i_param pretends not to see the
    # footprint (equal weights) -> benchmark for basis-risk comparison
    theta   = h @ W
    i_param = h @ np.full(T, 1.0 / T)

    # the sponsor's true loss. eta is the leftover noise inside a tier.
    # same [0,1] scale as theta
    eta = rng.lognormal(-0.5 * p["ETA_SIGMA"] ** 2, p["ETA_SIGMA"], (n_events, T))
    ls  = (E * h * eta).sum(axis=1) / E.sum()

    # the layer: nothing below attach, full notional above exhaust, linear between
    layer       = lambda x: p["N_NOTIONAL"] * np.clip((x - a) / (e - a), 0.0, 1.0)
    payout      = layer(theta)
    payout_par  = layer(i_param)
    ls_layer    = layer(ls)

    # only the worst event of the year counts. per-occurrence,
    # not aggregate
    yr_payout = np.zeros(p["N_YEARS"])
    np.maximum.at(yr_payout, year_id, payout)

    return dict(theta=theta, i_param=i_param, ls=ls, ls_layer=ls_layer,
                payout=payout, payout_par=payout_par, yr_payout=yr_payout)


# --- metrics ----------------------------------------------------------

def expected_loss(yr_payout, N):
    return yr_payout.mean() / N

def attachment_prob(yr_payout):
    return (yr_payout > 0).mean()

def tvar(yr_payout, N, rp):
    # average of the losses past the 1-in-rp quantile
    var  = np.quantile(yr_payout, 1 - 1 / rp)
    tail = yr_payout[yr_payout >= var]
    return (tail.mean() if len(tail) else var) / N

def conditional_variance(signal, loss, nbins=50):
    # E[Var(loss | signal)]: sort by the signal, bin it, average the within-bin
    # variance
    order = np.argsort(signal)
    bins  = np.array_split(loss[order], nbins)
    return sum(np.var(b) * len(b) for b in bins) / len(loss)


# --- run --------------------------------------------------------------

def main():
    p = P
    N = p["N_NOTIONAL"]

    s_t = simulate(p, copula="t",        seed=1)   # the real run
    s_g = simulate(p, copula="gaussian", seed=1)   # the optimistic benchmark

    # how much of the irreducible basis risk the footprint actually removes
    irr_par = conditional_variance(s_t["i_param"], s_t["ls"])
    irr_hyb = conditional_variance(s_t["theta"],   s_t["ls"])

    res = {
        "EL_pct":          100 * expected_loss(s_t["yr_payout"], N),
        "attach_prob_pct": 100 * attachment_prob(s_t["yr_payout"]),
        "TVaR_100_pct":    100 * tvar(s_t["yr_payout"], N, 100),
        "TVaR_250_pct":    100 * tvar(s_t["yr_payout"], N, 250),
        "EL_gauss_pct":    100 * expected_loss(s_g["yr_payout"], N),
        "TVaR100_gauss":   100 * tvar(s_g["yr_payout"], N, 100),
        "TVaR250_gauss":   100 * tvar(s_g["yr_payout"], N, 250),
        "sd_basis_hybrid": np.std(s_t["ls_layer"] - s_t["payout"]),
        "sd_basis_param":  np.std(s_t["ls_layer"] - s_t["payout_par"]),
        "delta_footprint": irr_par - irr_hyb,
        "pct_closed":      100 * (irr_par - irr_hyb) / irr_par,
    }
    res["spread_indic_pct"] = 5.0 * res["EL_pct"]   # rough EL multiple of 5, just a sanity check

    print("\nGalileo Re - base-case results")
    for k, v in res.items():
        print(f"  {k:18s}: {v:8.4f}")

    pd.Series(res).to_csv(f"{OUT}/galileo_re_results.csv", header=["value"])
    with open(f"{OUT}/galileo_re_results.json", "w") as f:
        json.dump({k: float(v) for k, v in res.items()}, f, indent=2)

    make_figures(s_t, s_g, irr_par, irr_hyb, N)
    print(f"\nResults and figures written to {OUT}")


# --- figures ----------------------------------------------------------

def make_figures(s_t, s_g, irr_par, irr_hyb, N):
    rng = np.random.default_rng(0)

    # exceedance curve: real run vs the gaussian benchmark
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, lab, c in [(s_t, "t-copula (tail-dependent)", "#1f5c8b"),
                      (s_g, "Gaussian copula ($\\lambda_U=0$)", "#c0392b")]:
        loss = np.sort(s["yr_payout"][s["yr_payout"] > 0] / N * 100)[::-1]
        ex   = np.arange(1, len(loss) + 1) / len(s["yr_payout"])
        ax.plot(loss, ex, color=c, lw=1.8, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Principal loss (% of notional)")
    ax.set_ylabel("Annual exceedance probability")
    ax.set_title("Galileo Re - exceedance probability curve")
    for rp in (100, 250):
        ax.axhline(1 / rp, ls="--", lw=.8, color="grey")
        ax.text(0.5, 1.1 / rp, f"1-in-{rp}", fontsize=8, color="grey")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_exceedance.png", dpi=150)

    # basis risk: hybrid vs parametric
    idx = rng.choice(len(s_t["theta"]), size=5000, replace=False)
    lim = max(s_t["payout"].max(), s_t["payout_par"].max(), s_t["ls_layer"].max())
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    for k, (col, ttl, pay) in enumerate([
            ("#1f5c8b", "Hybrid trigger (duration + footprint)", "payout"),
            ("#c0392b", "Pure parametric trigger (duration only)", "payout_par")]):
        ax[k].scatter(s_t[pay][idx], s_t["ls_layer"][idx], s=9, alpha=.28,
                      color=col, edgecolors="none")
        ax[k].plot([0, lim], [0, lim], ls="--", color="black", lw=1)
        ax[k].set_title(ttl, fontsize=11)
        ax[k].set_xlabel("Bond payout  (USD m)")
        ax[k].grid(alpha=.25)
        sd = np.std(s_t["ls_layer"] - s_t[pay])
        ax[k].text(0.04, 0.96, f"$\\sigma(L^S-P)$ = {sd:.1f}",
                   transform=ax[k].transAxes, va="top", fontsize=9.5, color=col)
    ax[0].set_ylabel("Sponsor layer loss $L^S$  (USD m)")
    ax[1].plot([], [], ls="--", color="black", lw=1, label="perfect match")
    ax[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Basis risk: the footprint modifier tightens the payout around the loss",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/fig_basisrisk.png", dpi=150)

    # the same thing as one number: how much of the gap the footprint closes
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["Pure parametric\n(duration only)", "Hybrid\n(duration+footprint)"],
           [irr_par, irr_hyb], color=["#c0392b", "#1f5c8b"])
    ax.set_ylabel("Irreducible basis risk  $E[\\mathrm{Var}(L^S\\,|\\,V)]$")
    ax.set_title(f"Footprint modifier closes {100*(irr_par-irr_hyb)/irr_par:.0f}% of the gap")
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_basis_reduction.png", dpi=150)


if __name__ == "__main__":
    main()
