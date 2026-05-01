"""
CLI Demo — exercise the full prediction pipeline from the terminal.
Usage:
    python scripts/demo.py
"""

import sys
import json
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import CoverPredictor, StaffScheduler, IngredientOrderer
from dataclasses import asdict


def divider(title=""):
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * w)


def run_demo():
    print("\n" + "═" * 70)
    print("  MISE EN PLACE — Restaurant Intelligence Demo")
    print("═" * 70)

    # ── Load model ──────────────────────────────────────────────────────────
    try:
        predictor = CoverPredictor.load()
        print(f"\n✓ Model loaded  |  CV MAE: {predictor.cv_score:.2f} covers"
              f"  |  Feedback events: {len(predictor.feedback_history)}")
    except FileNotFoundError:
        print("  Model not found. Train first: python scripts/train.py")
        sys.exit(1)

    scheduler = StaffScheduler()
    orderer = IngredientOrderer()

    # ── 1. Cover Forecast ───────────────────────────────────────────────────
    target_date = str(date.today() + timedelta(days=1))
    weather = "sunny"
    divider("1 · COVER FORECAST")
    print(f"  Date   : {target_date}")
    print(f"  Weather: {weather}\n")

    hourly_covers = {}
    print(f"  {'Hour':>5}  {'Predicted':>10}  {'Range':>18}  {'Conf':>6}")
    print(f"  {'────':>5}  {'─────────':>10}  {'──────────────────':>18}  {'────':>6}")
    for hour in range(11, 23):
        p = predictor.predict(target_date, hour, weather)
        hourly_covers[hour] = p.predicted_covers
        print(f"  {hour:>4}h  {p.predicted_covers:>10.1f}  "
              f"  [{p.lower_bound:>6.1f} – {p.upper_bound:>6.1f}]  {p.confidence*100:>5.0f}%")

    total = sum(hourly_covers.values())
    print(f"\n  Daily total : {total:.0f} covers")
    print(f"  Model stats : {predictor.get_feedback_stats()}")

    # ── 2. Staff Schedule ───────────────────────────────────────────────────
    divider("2 · STAFF SCHEDULE")
    requirements = scheduler.schedule(hourly_covers)
    summary = scheduler.daily_summary(requirements)

    print(f"\n  {'Role':20}  {'Station':12}  {'Peak':>5}  {'Hours':>6}  {'Cost':>8}")
    print(f"  {'────':20}  {'───────':12}  {'────':>5}  {'─────':>6}  {'────':>8}")
    for role in summary["by_role"]:
        print(f"  {role['role']:20}  {role['station']:12}  {role['peak_count']:>5}  "
              f"{role['total_hours']:>6}  £{role['total_cost']:>7.2f}")
    print(f"\n  Total labour cost : £{summary['total_labor_cost']:.2f}")
    print(f"  Total staff hours : {summary['total_staff_hours']}")

    # ── 3. Ingredient Orders ─────────────────────────────────────────────────
    divider("3 · INGREDIENT ORDERS (next 3 days)")
    daily_totals = {}
    for i in range(3):
        d_str = str(date.today() + timedelta(days=i))
        day_total = sum(
            predictor.predict(d_str, h, weather).predicted_covers
            for h in range(11, 23)
        )
        daily_totals[d_str] = day_total

    orders = orderer.calculate_orders(daily_totals)
    cost_summary = orderer.order_cost_summary(orders)

    print(f"\n  {'Ingredient':22}  {'Qty':>8}  {'Unit':>6}  {'Urgency':>8}  {'Cost':>8}")
    print(f"  {'──────────':22}  {'───':>8}  {'────':>6}  {'───────':>8}  {'────':>8}")
    for o in orders[:12]:
        print(f"  {o.ingredient.replace('_',' '):22}  {o.quantity_to_order:>8.1f}  "
              f"{o.unit:>6}  {o.urgency:>8}  £{o.cost:>7.2f}")

    print(f"\n  Total food cost : £{cost_summary['total_food_cost']:.2f}")
    for cat, cost in cost_summary["by_category"].items():
        print(f"    {cat:15}: £{cost:.2f}")

    # ── 4. Simulate Feedback ─────────────────────────────────────────────────
    divider("4 · FEEDBACK LOOP SIMULATION")
    yesterday = str(date.today() - timedelta(days=1))
    predicted_total = 450.0
    actual_total = 360.0  # Rainy day, fewer people came
    print(f"\n  Scenario : Predicted {predicted_total:.0f} covers, got {actual_total:.0f}")
    print(f"  Reason   : Unexpected heavy rain")
    print(f"\n  Applying correction to model...")

    for hour in [12, 13, 19, 20]:
        pred = predictor.predict(yesterday, hour, "rainy")
        actual_hour = pred.predicted_covers * (actual_total / predicted_total)
        predictor.apply_feedback(
            date_str=yesterday, hour=hour, weather="rainy",
            predicted=pred.predicted_covers, actual=actual_hour,
            note="Heavy rain - far fewer covers than expected"
        )

    print(f"\n  After correction:")
    stats = predictor.get_feedback_stats()
    for k, v in stats.items():
        print(f"    {k:20}: {v}")

    print(f"\n  Re-predicting same date with 'rainy' weather:")
    for hour in [12, 19, 20]:
        p = predictor.predict(yesterday, hour, "rainy")
        print(f"    {hour}:00 → {p.predicted_covers:.1f} covers "
              f"(base: {p.factors['base_prediction']:.1f}, "
              f"correction: {p.factors['correction']:+.1f})")

    divider()
    print("\n  Demo complete. Launch the web dashboard: python api/app.py\n")


if __name__ == "__main__":
    run_demo()
