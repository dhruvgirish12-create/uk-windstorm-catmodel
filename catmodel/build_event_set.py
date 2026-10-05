"""
UK Windstorm Catastrophe Model — Stage 1: Stochastic Event Set

Builds a synthetic stochastic event set for UK extratropical windstorm and
persists it to SQLite, mirroring the structure a commercial cat model uses:
each event carries an ID, a rate (annual frequency), a severity parameter,
and a regional footprint.

This is a teaching/portfolio model built on published-style parameters, not
a calibrated commercial model. Figures are illustrative.
"""

import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

RNG = np.random.default_rng(42)
DB = Path(__file__).parent / "catmodel.db"

# ---------------------------------------------------------------------------
# Portfolio: a notional UK property book split across five regions.
# Total insured value (TIV) in £m.
# ---------------------------------------------------------------------------
REGIONS = pd.DataFrame({
    "region":      ["Scotland", "North England", "Midlands", "South England", "Wales"],
    "tiv_m":       [820.0,       1450.0,          1310.0,     2600.0,          470.0],
    # Relative windstorm hazard — north and west of the UK are more exposed to
    # North Atlantic extratropical cyclones than the south east.
    "hazard_mult": [1.35,        1.15,            0.90,       0.75,            1.25],
})

N_EVENTS = 2500          # size of the stochastic event catalogue
LAMBDA_ANNUAL = 4.2      # expected number of landfalling windstorms per year


def build_event_set(n_events: int = N_EVENTS) -> pd.DataFrame:
    """Generate the stochastic event catalogue.

    Each event gets:
      - a severity index drawn from a lognormal (heavy right tail, as storm
        intensity distributions have),
      - an annual rate, so the catalogue rates sum to LAMBDA_ANNUAL,
      - a primary region, weighted by relative hazard.
    """
    severity = RNG.lognormal(mean=0.0, sigma=0.85, size=n_events)

    region_weights = REGIONS["hazard_mult"] / REGIONS["hazard_mult"].sum()
    primary_region = RNG.choice(REGIONS["region"], size=n_events, p=region_weights)

    # Rarer (more severe) events get lower rates: rate declines with severity.
    raw_rate = 1.0 / (severity ** 1.25)
    rate = raw_rate / raw_rate.sum() * LAMBDA_ANNUAL

    return pd.DataFrame({
        "event_id": np.arange(1, n_events + 1),
        "primary_region": primary_region,
        "severity_index": np.round(severity, 4),
        "annual_rate": rate,
    })


def write_db(events: pd.DataFrame, regions: pd.DataFrame, db_path: Path = DB) -> None:
    """Persist the event set and exposure to SQLite."""
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    events.to_sql("events", con, index=False)
    regions.to_sql("exposure", con, index=False)
    con.execute("CREATE INDEX idx_events_region ON events(primary_region)")
    con.commit()
    con.close()


if __name__ == "__main__":
    events = build_event_set()
    write_db(events, REGIONS)

    print(f"Event catalogue: {len(events):,} events")
    print(f"Catalogue annual rate: {events['annual_rate'].sum():.2f} events/year")
    print(f"Portfolio TIV: £{REGIONS['tiv_m'].sum():,.0f}m")
    print(f"\nWritten to {DB}")
    print("\nEvents by region (SQL):")

    con = sqlite3.connect(DB)
    q = """
        SELECT e.primary_region           AS region,
               COUNT(*)                   AS n_events,
               ROUND(SUM(e.annual_rate),3) AS annual_rate,
               ROUND(MAX(e.severity_index),2) AS max_severity,
               x.tiv_m
        FROM events e
        JOIN exposure x ON x.region = e.primary_region
        GROUP BY e.primary_region
        ORDER BY annual_rate DESC
    """
    print(pd.read_sql(q, con).to_string(index=False))
    con.close()
