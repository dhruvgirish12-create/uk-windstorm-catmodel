"""
UK Windstorm Catastrophe Model — Stage 2: Simulation, EP Curve, Reinsurance

Runs a 10,000-year Monte Carlo simulation over the stochastic event set,
applies a damage function and policy terms to produce ground-up and insured
losses, builds OEP and AEP curves, and tests a reinsurance structure.

Outputs: AAL, PML at standard return periods, and the loss ceded to a
£50m xs £50m layer.
"""

import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(2026)
HERE = Path(__file__).parent
DB = HERE / "catmodel.db"

N_YEARS = 10_000
DEDUCTIBLE_PCT = 0.02        # 2% of TIV retained by the insured per event
LAYER_ATTACH_M = 50.0        # reinsurance attaches at £50m
LAYER_LIMIT_M = 50.0         # £50m of cover above that


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(DB)
    events = pd.read_sql("SELECT * FROM events", con)
    exposure = pd.read_sql("SELECT * FROM exposure", con)
    con.close()
    return events, exposure


# ---------------------------------------------------------------------------
# Damage function
# ---------------------------------------------------------------------------
def mean_damage_ratio(severity: np.ndarray, hazard_mult: np.ndarray) -> np.ndarray:
    """Vulnerability curve mapping storm severity to a mean damage ratio.

    Uses a saturating exponential: damage rises steeply with severity then
    flattens, which is the shape of observed windstorm vulnerability curves.
    Capped at 35% — a windstorm damages roofs and cladding, it does not
    destroy the whole building stock the way an earthquake can.
    """
    intensity = severity * hazard_mult
    mdr = 0.35 * (1.0 - np.exp(-0.055 * intensity ** 1.45))
    return np.clip(mdr, 0.0, 0.35)


def simulate_year_losses(events: pd.DataFrame, exposure: pd.DataFrame,
                         n_years: int = N_YEARS) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo over the event catalogue.

    Returns:
      annual_losses : total insured loss per simulated year (AEP basis)
      event_losses  : largest single-event insured loss per year (OEP basis)
    """
    exp_map = exposure.set_index("region")
    rates = events["annual_rate"].to_numpy()
    total_rate = rates.sum()
    probs = rates / total_rate

    sev = events["severity_index"].to_numpy()
    regions = events["primary_region"].to_numpy()
    tiv = exp_map.loc[regions, "tiv_m"].to_numpy()
    hmult = exp_map.loc[regions, "hazard_mult"].to_numpy()

    # Pre-compute each event's expected insured loss
    mdr = mean_damage_ratio(sev, hmult)

    annual_losses = np.zeros(n_years)
    max_event_losses = np.zeros(n_years)

    # Number of events per year ~ Poisson(total_rate)
    n_events_per_year = RNG.poisson(total_rate, size=n_years)

    for yr in range(n_years):
        k = n_events_per_year[yr]
        if k == 0:
            continue
        idx = RNG.choice(len(events), size=k, p=probs)

        # Secondary uncertainty: realised damage varies around the mean ratio
        realised_mdr = np.clip(
            mdr[idx] * RNG.lognormal(mean=-0.12, sigma=0.5, size=k), 0.0, 0.60
        )
        ground_up = realised_mdr * tiv[idx]

        # Policy terms: insured retains a per-event deductible
        deductible = DEDUCTIBLE_PCT * tiv[idx]
        insured = np.maximum(ground_up - deductible, 0.0)

        annual_losses[yr] = insured.sum()
        max_event_losses[yr] = insured.max()

    return annual_losses, max_event_losses


# ---------------------------------------------------------------------------
# Exceedance probability
# ---------------------------------------------------------------------------
def ep_curve(losses: np.ndarray) -> pd.DataFrame:
    """Build an exceedance probability curve from simulated losses."""
    s = np.sort(losses)[::-1]
    n = len(s)
    exceed_prob = np.arange(1, n + 1) / n
    return pd.DataFrame({
        "loss_m": s,
        "exceedance_prob": exceed_prob,
        "return_period": 1.0 / exceed_prob,
    })


def pml(curve: pd.DataFrame, return_period: float) -> float:
    """Probable Maximum Loss at a given return period, by interpolation."""
    c = curve.sort_values("return_period")
    return float(np.interp(return_period, c["return_period"], c["loss_m"]))


# ---------------------------------------------------------------------------
# Reinsurance
# ---------------------------------------------------------------------------
def apply_layer(losses: np.ndarray, attach: float, limit: float) -> np.ndarray:
    """Loss ceded to an excess-of-loss layer: limit xs attach."""
    return np.clip(losses - attach, 0.0, limit)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    events, exposure = load()
    tiv = exposure["tiv_m"].sum()

    annual, occurrence = simulate_year_losses(events, exposure)

    aep = ep_curve(annual)
    oep = ep_curve(occurrence)

    aal = annual.mean()
    ceded = apply_layer(annual, LAYER_ATTACH_M, LAYER_LIMIT_M)
    retained = annual - ceded

    layer_aal = ceded.mean()
    attach_prob = (annual > LAYER_ATTACH_M).mean()
    exhaust_prob = (annual > LAYER_ATTACH_M + LAYER_LIMIT_M).mean()

    print("=" * 66)
    print("UK WINDSTORM PORTFOLIO — CATASTROPHE MODEL RESULTS")
    print("=" * 66)
    print(f"Total insured value          £{tiv:,.0f}m")
    print(f"Simulated years              {N_YEARS:,}")
    print(f"Event catalogue              {len(events):,} events")
    print()
    print(f"Average Annual Loss (AAL)    £{aal:,.2f}m   ({aal / tiv * 10000:.1f} bps of TIV)")
    print()
    print("Aggregate EP (AEP) — annual portfolio loss")
    for rp in (10, 25, 50, 100, 200, 250):
        print(f"  1-in-{rp:<4} year          £{pml(aep, rp):,.1f}m")
    print()
    print("Occurrence EP (OEP) — largest single event")
    for rp in (10, 25, 50, 100, 200, 250):
        print(f"  1-in-{rp:<4} year          £{pml(oep, rp):,.1f}m")
    print()
    print(f"REINSURANCE: £{LAYER_LIMIT_M:,.0f}m xs £{LAYER_ATTACH_M:,.0f}m (aggregate basis)")
    print(f"  Probability layer attaches   {attach_prob * 100:,.2f}%  "
          f"(1-in-{1/attach_prob:,.0f} years)" if attach_prob > 0 else "  never attaches")
    print(f"  Probability layer exhausts   {exhaust_prob * 100:,.2f}%  "
          f"(1-in-{1/exhaust_prob:,.0f} years)" if exhaust_prob > 0 else "  never exhausts")
    print(f"  Expected loss to layer       £{layer_aal:,.2f}m")
    print(f"  Technical rate on line       {layer_aal / LAYER_LIMIT_M * 100:,.2f}%")
    print()
    print(f"  Gross 1-in-100               £{pml(aep, 100):,.1f}m")
    print(f"  Net of layer 1-in-100        £{pml(ep_curve(retained), 100):,.1f}m")
    print(f"  Tail reduction at 1-in-100   "
          f"{(1 - pml(ep_curve(retained), 100) / pml(aep, 100)) * 100:,.1f}%")
    print("=" * 66)

    # ---- Chart ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    g = aep[aep["return_period"] >= 2]
    r = ep_curve(retained)
    r = r[r["return_period"] >= 2]

    ax.plot(g["return_period"], g["loss_m"], lw=2.0, color="#1f3a5f", label="Gross loss")
    ax.plot(r["return_period"], r["loss_m"], lw=2.0, color="#c2703a",
            label=f"Net of £{LAYER_LIMIT_M:.0f}m xs £{LAYER_ATTACH_M:.0f}m")
    ax.axhline(LAYER_ATTACH_M, color="#999999", ls="--", lw=1.0)
    ax.axhline(LAYER_ATTACH_M + LAYER_LIMIT_M, color="#999999", ls="--", lw=1.0)
    ax.fill_between(g["return_period"], LAYER_ATTACH_M, LAYER_ATTACH_M + LAYER_LIMIT_M,
                    color="#c2703a", alpha=0.10, label="Reinsurance layer")

    ax.set_xscale("log")
    ax.set_xlim(2, 1000)
    ax.set_xlabel("Return period (years, log scale)")
    ax.set_ylabel("Annual portfolio loss (£m)")
    ax.set_title("UK Windstorm Portfolio — Aggregate Exceedance Probability Curve",
                 fontsize=12, pad=12)
    ax.set_xticks([2, 5, 10, 25, 50, 100, 250, 500, 1000])
    ax.set_xticklabels(["2", "5", "10", "25", "50", "100", "250", "500", "1000"])
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "ep_curve.png", dpi=170)
    print(f"Chart written to {HERE / 'ep_curve.png'}")

    # ---- Persist results --------------------------------------------------
    con = sqlite3.connect(DB)
    pd.DataFrame({"year": np.arange(1, N_YEARS + 1),
                  "annual_loss_m": annual,
                  "max_event_loss_m": occurrence,
                  "ceded_m": ceded,
                  "retained_m": retained}).to_sql(
        "simulated_years", con, index=False, if_exists="replace")
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
