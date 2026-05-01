"""
Generate a realistic synthetic dataset for the restaurant forecasting system.
Creates 2 years of hourly cover data with realistic patterns.

Usage:
    python scripts/generate_data.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"

np.random.seed(42)

WEATHER_POOL = {
    1: ["rainy", "cloudy", "sunny"],       # Jan
    2: ["rainy", "cloudy", "sunny"],
    3: ["cloudy", "sunny", "rainy"],
    4: ["sunny", "cloudy", "rainy"],
    5: ["sunny", "sunny", "cloudy"],
    6: ["sunny", "sunny", "cloudy"],       # Jun
    7: ["sunny", "sunny", "sunny"],
    8: ["sunny", "sunny", "cloudy"],
    9: ["sunny", "cloudy", "rainy"],
    10: ["cloudy", "rainy", "sunny"],
    11: ["rainy", "cloudy", "sunny"],
    12: ["rainy", "cloudy", "sunny"],      # Dec
}

WEATHER_IMPACT = {"sunny": 1.0, "cloudy": 0.88, "rainy": 0.68, "stormy": 0.45}

DAY_BASE = {
    "Monday": 55, "Tuesday": 58, "Wednesday": 64,
    "Thursday": 72, "Friday": 98, "Saturday": 118, "Sunday": 105
}

HOUR_DIST = {
    11: 0.35, 12: 0.80, 13: 1.00, 14: 0.72, 15: 0.38,
    16: 0.22, 17: 0.48, 18: 0.78, 19: 1.00, 20: 0.96, 21: 0.70, 22: 0.38
}

SPECIAL_DATES = {
    "01-01": ("New Year's Day", 1.60),
    "02-14": ("Valentine's Day", 1.75),
    "03-17": ("St. Patrick's Day", 1.30),
    "05-12": ("Mother's Day", 1.55),
    "06-16": ("Father's Day", 1.35),
    "10-31": ("Halloween", 1.25),
    "12-24": ("Christmas Eve", 1.50),
    "12-25": ("Christmas Day", 0.60),
    "12-31": ("New Year's Eve", 1.80),
}


def is_special(d: date):
    key = d.strftime("%m-%d")
    return SPECIAL_DATES.get(key, (None, 1.0))


def generate_covers_for_day(d: date, weather: str) -> list[dict]:
    rows = []
    day_name = d.strftime("%A")
    base = DAY_BASE.get(day_name, 65)
    w_impact = WEATHER_IMPACT[weather]
    special_name, special_mult = is_special(d)

    # Monthly seasonal factor
    month_factors = {
        1: 0.82, 2: 0.88, 3: 0.92, 4: 0.96, 5: 1.00, 6: 1.05,
        7: 1.08, 8: 1.06, 9: 1.02, 10: 0.98, 11: 0.94, 12: 1.10
    }
    seasonal = month_factors.get(d.month, 1.0)

    # Year-over-year growth (restaurant is growing)
    year_growth = 1.0 + (d.year - 2024) * 0.06

    for hour, curve in HOUR_DIST.items():
        expected = base * curve * w_impact * special_mult * seasonal * year_growth
        noise = np.random.normal(0, expected * 0.08)
        covers = max(0, int(round(expected + noise)))

        rows.append({
            "date": d.isoformat(),
            "day_of_week": day_name,
            "hour": hour,
            "covers": covers,
            "weather": weather,
            "is_holiday": special_name is not None,
            "is_special_event": special_name is not None,
            "notes": special_name or ""
        })
    return rows


def main():
    print("Generating synthetic restaurant dataset (2024-01-01 to 2025-12-31)...")
    records = []
    start = date(2024, 1, 1)
    end = date(2025, 12, 31)

    d = start
    while d <= end:
        weather_pool = WEATHER_POOL[d.month]
        weather = np.random.choice(weather_pool, p=[0.5, 0.35, 0.15])
        records.extend(generate_covers_for_day(d, weather))
        d += timedelta(days=1)

    df = pd.DataFrame(records)
    out = DATA_DIR / "historical_covers.csv"
    df.to_csv(out, index=False)
    print(f"✓ Saved {len(df)} records to {out}")
    print(f"  Date range : {df['date'].min()} → {df['date'].max()}")
    print(f"  Covers     : min={df['covers'].min()}, max={df['covers'].max()}, "
          f"mean={df['covers'].mean():.1f}")
    print(f"  Total days : {df['date'].nunique()}")

    # Generate corresponding feedback log
    feedback = []
    sample_dates = df[df["date"] < "2025-06-01"]["date"].unique()
    sample = np.random.choice(sample_dates, size=min(80, len(sample_dates)), replace=False)
    fid = 1
    for d_str in sorted(sample):
        day_df = df[df["date"] == d_str]
        total_actual = day_df["covers"].sum()
        # Simulate predicted was close but not perfect
        total_predicted = total_actual * np.random.uniform(0.85, 1.18)
        feedback.append({
            "feedback_id": fid,
            "date": d_str,
            "predicted_covers": round(total_predicted),
            "actual_covers": total_actual,
            "weather_actual": day_df["weather"].iloc[0],
            "notes": day_df["notes"].iloc[0],
            "adjustment_factor": round(total_actual / max(total_predicted, 1), 3),
            "recorded_at": f"{d_str} 23:30:00"
        })
        fid += 1

    fb_df = pd.DataFrame(feedback)
    fb_out = DATA_DIR / "feedback_log.csv"
    fb_df.to_csv(fb_out, index=False)
    print(f"✓ Saved {len(fb_df)} feedback records to {fb_out}")


if __name__ == "__main__":
    main()
