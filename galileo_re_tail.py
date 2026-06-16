"""
Tail stress tests.

I move one tail-driving assumption at a time and watch what the extreme quantiles
do. The question I'm after: is the tail fragile to the dependence, to the
severity, or to event clustering?
"""
import copy, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from galileo_re_model import P, simulate, expected_loss, tvar, attachment_prob, OUT

N = P["N_NOTIONAL"]

def metrics(s):
    # the four numbers I care about for each scenario
    yp = s["yr_payout"]
    return dict(EL=100*expected_loss(yp, N),
                AP=100*attachment_prob(yp),
                T100=100*tvar(yp, N, 100),
                T250=100*tvar(yp, N, 250))

def run(label, **over):
    # copy the base params, override the one thing being stressed, simulate
    p = copy.deepcopy(P)
    cop = over.pop("copula", "t")
    clu = over.pop("clustered", False)
    p.update(over)
    return label, metrics(simulate(p, copula=cop, clustered=clu, seed=1))

# one row per stress, base case first
scenarios = [
    run("Base case (df=3)"),
    run("Heavier tail dep. (df=2)",      COPULA_DF=2.0),
    run("Lighter tail dep. (df=5)",      COPULA_DF=5.0),
    run("No tail dep. (Gaussian)",       copula="gaussian"),
    run("Lighter severity (sig=1.0)",    LN_SIGMA=1.0),
    run("Heavier severity (sig=1.3)",    LN_SIGMA=1.3),
    run("Event clustering",              clustered=True),
]

df = pd.DataFrame({lab: m for lab, m in scenarios}).T
df.columns = ["EL (%)", "Attach. prob. (%)", "TVaR-100 (%)", "TVaR-250 (%)"]
df = df.round(2)
print(df.to_string())
df.to_csv(f"{OUT}/galileo_re_tail_stress.csv")

# figure: exceedance curves as the copula tail gets heavier. gaussian sits
# well below the otherss
fig, ax = plt.subplots(figsize=(7.2, 4.6))
curves = [("df = 2", dict(COPULA_DF=2.0), "t", "#7b1f1f"),
          ("df = 3 (base)", dict(), "t", "#1f5c8b"),
          ("df = 5", dict(COPULA_DF=5.0), "t", "#2e8b57"),
          ("Gaussian", dict(), "gaussian", "#c0392b")]
for lab, over, cop, c in curves:
    p = copy.deepcopy(P); p.update(over)
    s = simulate(p, copula=cop, seed=1)
    loss = np.sort(s["yr_payout"][s["yr_payout"] > 0] / N * 100)[::-1]
    ex = np.arange(1, len(loss)+1) / len(s["yr_payout"])
    ax.plot(loss, ex, lw=1.7, color=c, label=lab)
ax.set_yscale("log")
ax.set_xlabel("Principal loss (% of notional)")
ax.set_ylabel("Annual exceedance probability")
ax.set_title("Tail sensitivity to the copula degrees of freedom")
for rp in (100, 250):
    ax.axhline(1/rp, ls="--", lw=.8, color="grey")
    ax.text(0.5, 1.1/rp, f"1-in-{rp}", fontsize=8, color="grey")
ax.legend(frameon=False, fontsize=9, title="tail dependence"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_tail_sensitivity.png", dpi=150)
print("\nsaved table + figure")
