# Galileo Re — cyber catastrophe bond model

Code for my master's thesis on transferring systemic cyber risk through a
catastrophe bond.

The random seed is fixed, so every run reproduces the same numbers and figures.

## Requirements

Python 3.10+ and a few standard packages:

```bash
pip install numpy scipy pandas matplotlib
```

## The scripts

Run them from the repository folder. `galileo_re_model.py` must be present for the
others, since they import from it.

| Script | What it does | Thesis chapter |
| `galileo_re_model.py` | Monte Carlo engine: frequency, propagation, severity. Produces base-case loss numbers and basis-risk figures. |
| `galileo_re_tail.py` | Tail stress tests: perturbs the copula, the severity, and event clustering to see how robust the tail is. |
| `galileo_re_pricing.py` | Pricing: Wang transform for the risk load, robust pricing over an ambiguity set, and the resilience covenant. |
| `galileo_re_sensitivity.py` | Sensitivity analysis: basis-risk curve, the tornado, and the robustness of the covenant finding. |

## How to run

```bash
python galileo_re_model.py        # base case + basis-risk figures
python galileo_re_tail.py         # tail stress tests
python galileo_re_pricing.py      # spread, decomposition, covenant
python galileo_re_sensitivity.py  # tornado + sensitivity
```

Each script prints its results to the console and writes its tables (`.csv`) and
figures (`.png`) to an output folder. The model runs on two million simulated
years by default; the sensitivity script uses one million, since it does many
runs and only the size of the movements matters there.

