"""
Restaurant Forecasting REST API
Endpoints for predictions, feedback, staff scheduling, and ingredient ordering.
"""

import sys
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from dataclasses import asdict

from flask import Flask, request, jsonify, render_template
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.forecaster import CoverPredictor, StaffScheduler, IngredientOrderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# ── Lazy-load singletons ──────────────────────────────────────────────────────

_predictor: CoverPredictor = None
_scheduler: StaffScheduler = None
_orderer: IngredientOrderer = None


def get_predictor() -> CoverPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CoverPredictor.load()
    return _predictor


def get_scheduler() -> StaffScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = StaffScheduler()
    return _scheduler


def get_orderer() -> IngredientOrderer:
    global _orderer
    if _orderer is None:
        _orderer = IngredientOrderer()
    return _orderer


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict/covers", methods=["POST"])
def predict_covers():
    """
    Predict hourly covers for a date.
    Body: { date, weather, is_holiday, is_special_event }
    Returns: { hourly: [...], daily_total, summary }
    """
    data = request.json or {}
    target_date = data.get("date", str(date.today() + timedelta(days=1)))
    weather = data.get("weather", "sunny")
    is_holiday = bool(data.get("is_holiday", False))
    is_special_event = bool(data.get("is_special_event", False))

    predictor = get_predictor()
    service_hours = list(range(11, 23))

    hourly = []
    for hour in service_hours:
        pred = predictor.predict(
            target_date, hour, weather, is_holiday, is_special_event
        )
        hourly.append(asdict(pred))

    total = sum(h["predicted_covers"] for h in hourly)
    lower = sum(h["lower_bound"] for h in hourly)
    upper = sum(h["upper_bound"] for h in hourly)
    avg_conf = sum(h["confidence"] for h in hourly) / len(hourly)

    return jsonify({
        "date": target_date,
        "weather": weather,
        "hourly": hourly,
        "daily_total": round(total, 1),
        "daily_lower": round(lower, 1),
        "daily_upper": round(upper, 1),
        "confidence": round(avg_conf, 2),
        "feedback_stats": predictor.get_feedback_stats()
    })


@app.route("/api/predict/staff", methods=["POST"])
def predict_staff():
    """
    Predict staffing needs given cover predictions.
    Body: { date, weather, is_holiday, is_special_event }
    Returns: { schedule_by_hour, daily_summary }
    """
    data = request.json or {}
    target_date = data.get("date", str(date.today() + timedelta(days=1)))
    weather = data.get("weather", "sunny")
    is_holiday = bool(data.get("is_holiday", False))
    is_special_event = bool(data.get("is_special_event", False))

    predictor = get_predictor()
    scheduler = get_scheduler()
    service_hours = list(range(11, 23))

    hourly_covers = {}
    for hour in service_hours:
        pred = predictor.predict(target_date, hour, weather, is_holiday, is_special_event)
        hourly_covers[hour] = pred.predicted_covers

    requirements = scheduler.schedule(hourly_covers)
    daily_summary = scheduler.daily_summary(requirements)

    # Group hourly schedule
    hourly_schedule = {}
    for r in requirements:
        if r.hour not in hourly_schedule:
            hourly_schedule[r.hour] = []
        hourly_schedule[r.hour].append(asdict(r))

    return jsonify({
        "date": target_date,
        "hourly_covers": hourly_covers,
        "hourly_schedule": hourly_schedule,
        "daily_summary": daily_summary
    })


@app.route("/api/predict/ingredients", methods=["POST"])
def predict_ingredients():
    """
    Calculate ingredient orders for upcoming days.
    Body: {
        days_ahead: 3,
        start_date: "2024-03-20",   # optional
        weather_forecast: {"2024-03-20": "sunny", ...},
        current_stock: {"chicken_breast": 5.0, ...}   # optional
    }
    """
    data = request.json or {}
    days_ahead = int(data.get("days_ahead", 3))
    start_date = data.get("start_date", str(date.today()))
    weather_forecast = data.get("weather_forecast", {})
    current_stock = data.get("current_stock", {})

    predictor = get_predictor()
    orderer = get_orderer()

    # Get daily cover totals for planning window
    daily_totals = {}
    d = pd.to_datetime(start_date)
    for i in range(days_ahead + 3):  # +3 for lead-time buffer
        d_str = (d + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        weather = weather_forecast.get(d_str, "sunny")
        day_total = 0
        for hour in range(11, 23):
            pred = predictor.predict(d_str, hour, weather)
            day_total += pred.predicted_covers
        daily_totals[d_str] = day_total

    orders = orderer.calculate_orders(daily_totals, current_stock, start_date)
    cost_summary = orderer.order_cost_summary(orders)

    return jsonify({
        "order_date": start_date,
        "planning_window_days": days_ahead,
        "expected_covers": daily_totals,
        "orders": [asdict(o) for o in orders],
        "cost_summary": cost_summary
    })


@app.route("/api/predict/full", methods=["POST"])
def predict_full():
    """
    Full prediction: covers + staff + ingredients in one call.
    Body: { date, weather, is_holiday, is_special_event }
    """
    data = request.json or {}
    target_date = data.get("date", str(date.today() + timedelta(days=1)))
    weather = data.get("weather", "sunny")
    is_holiday = bool(data.get("is_holiday", False))
    is_special_event = bool(data.get("is_special_event", False))

    predictor = get_predictor()
    scheduler = get_scheduler()
    orderer = get_orderer()

    service_hours = list(range(11, 23))
    hourly_covers = {}
    hourly_preds = []

    for hour in service_hours:
        pred = predictor.predict(target_date, hour, weather, is_holiday, is_special_event)
        hourly_covers[hour] = pred.predicted_covers
        hourly_preds.append(asdict(pred))

    total_covers = sum(hourly_covers.values())
    requirements = scheduler.schedule(hourly_covers)
    staff_summary = scheduler.daily_summary(requirements)

    daily_totals = {target_date: total_covers}
    orders = orderer.calculate_orders(daily_totals, order_date=target_date)
    cost_summary = orderer.order_cost_summary(orders)

    return jsonify({
        "date": target_date,
        "weather": weather,
        "is_holiday": is_holiday,
        "is_special_event": is_special_event,
        "covers": {
            "hourly": hourly_preds,
            "daily_total": round(total_covers, 1),
            "confidence": round(sum(h["confidence"] for h in hourly_preds) / len(hourly_preds), 2)
        },
        "staff": staff_summary,
        "ingredients": {
            "orders": [asdict(o) for o in orders[:10]],  # Top 10 for overview
            "cost_summary": cost_summary
        },
        "model_health": predictor.get_feedback_stats()
    })


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """
    Manager feedback to update model.
    Body: {
        date: "2024-03-20",
        predicted_covers: 120,
        actual_covers: 85,
        weather: "rainy",
        is_holiday: false,
        is_special_event: false,
        note: "Unexpected heavy rain all day"
    }
    """
    data = request.json or {}
    required = ["date", "predicted_covers", "actual_covers"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    predictor = get_predictor()

    # Apply feedback for each service hour proportionally
    target_date = data["date"]
    predicted_total = float(data["predicted_covers"])
    actual_total = float(data["actual_covers"])
    weather = data.get("weather", "sunny")
    is_holiday = bool(data.get("is_holiday", False))
    is_special_event = bool(data.get("is_special_event", False))
    note = data.get("note", "")

    # Apply correction at peak hours (19:00) as representative
    for hour in [12, 13, 19, 20]:
        pred_hour = predictor.predict(target_date, hour, weather, is_holiday, is_special_event)
        scale = actual_total / max(predicted_total, 1)
        actual_hour = pred_hour.predicted_covers * scale

        predictor.apply_feedback(
            date_str=target_date,
            hour=hour,
            weather=weather,
            predicted=pred_hour.predicted_covers,
            actual=actual_hour,
            is_holiday=is_holiday,
            is_special_event=is_special_event,
            note=note
        )

    # Log to CSV
    log_path = Path(__file__).parent.parent / "data" / "feedback_log.csv"
    new_row = {
        "feedback_id": len(predictor.feedback_history),
        "date": target_date,
        "predicted_covers": predicted_total,
        "actual_covers": actual_total,
        "weather_actual": weather,
        "notes": note,
        "adjustment_factor": round(actual_total / max(predicted_total, 1), 3),
        "recorded_at": pd.Timestamp.now().isoformat()
    }
    if log_path.exists():
        fb_df = pd.read_csv(log_path)
    else:
        fb_df = pd.DataFrame()
    fb_df = pd.concat([fb_df, pd.DataFrame([new_row])], ignore_index=True)
    fb_df.to_csv(log_path, index=False)

    stats = predictor.get_feedback_stats()
    return jsonify({
        "status": "success",
        "message": f"Correction applied. Model has now learned from {stats['count']} feedback events.",
        "feedback_stats": stats,
        "adjustment_factor": round(actual_total / max(predicted_total, 1), 3)
    })


@app.route("/api/feedback/history", methods=["GET"])
def feedback_history():
    predictor = get_predictor()
    return jsonify({
        "history": predictor.feedback_history[-50:],  # last 50
        "stats": predictor.get_feedback_stats()
    })


@app.route("/api/model/status", methods=["GET"])
def model_status():
    predictor = get_predictor()
    return jsonify({
        "is_trained": predictor.is_trained,
        "cv_mae": predictor.cv_score,
        "correction_fitted": predictor.correction_fitted,
        "feedback_stats": predictor.get_feedback_stats(),
        "top_features": sorted(
            predictor.feature_importance.items(),
            key=lambda x: x[1], reverse=True
        )[:8]
    })


if __name__ == "__main__":
    print("Starting Restaurant Forecasting API on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
