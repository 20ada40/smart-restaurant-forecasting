# Mise en Place — Restaurant Intelligence System

> Predict covers, staff, and ingredients. Learn from every correction.

---

## Overview

A full-stack restaurant forecasting system that solves the over/under-resourcing problem through a **feedback loop ML architecture**:

1. **Cover Prediction** — Hourly customer count forecasting using Gradient Boosting
2. **Staff Scheduling** — Role-by-role staffing derived from cover predictions
3. **Ingredient Ordering** — Smart orders accounting for shelf life and lead times
4. **Online Learning** — SGD-based correction model that updates every time a manager reports actuals

---

## Architecture

```
historical_covers.csv
         │
         ▼
  CoverPredictor          ┌─────────────────────────────┐
  ┌──────────────┐        │  Two-layer prediction:       │
  │ Base Model   │◄───────┤  1. GradientBoosting (batch) │
  │ (GBR)        │        │  2. SGDRegressor (online)    │
  └──────┬───────┘        └─────────────────────────────┘
         │
         ▼
  Hourly Cover Predictions
         │
    ┌────┴────┐
    ▼         ▼
StaffScheduler  IngredientOrderer
    │               │
    ▼               ▼
Role counts     Order quantities
(by hour)       (with lead times)
         
Manager Feedback ──► SGDRegressor.partial_fit()
                           │
                           ▼
                    Correction δ added to
                    every future prediction
```

### Why Two Models?

| Layer | Algorithm | Role |
|-------|-----------|------|
| Base | Gradient Boosting | Captures complex non-linear patterns from historical data (seasonality, events, weather interactions) |
| Correction | SGD Regressor | Learns the *residual* — the gap between base predictions and reality — from manager feedback, online/incrementally |

This means: the base model is retrained periodically (weekly/monthly), while the correction model updates **instantly** after each feedback event.

---

## Features

### Engineered Features
- Day-of-week weight (Friday = 1.0 baseline)
- Hour curve (lunch and dinner peaks)
- Weather score (sunny=1.0 → rainy=0.65 → stormy=0.45)
- Sine/cosine encoding of month and hour (captures seasonality)
- Interaction features: `weekend × weather`, `event × hour`, `day × hour`
- Holiday and special event flags

### Ingredient Ordering Logic
```
qty_to_order = max(0, expected_usage × 1.15_buffer − on_hand)
```
- **Window**: lead_time_days → lead_time + shelf_life_days
- **Urgency**: critical (shelf ≤ 3d or lead ≥ shelf), normal (shelf ≤ 7d), buffer
- **Min order**: rounds up to supplier minimum quantities

### Feedback Loop
1. Manager submits: predicted=120, actual=85, weather=rainy, note="Downpour"
2. System computes residual: -35 covers
3. `SGDRegressor.partial_fit([features], [-35])` — model updates weights
4. Next rainy prediction pulls down by the learned correction
5. Stats tracked: MAE, bias, convergence over time

---

## Project Structure

```
restaurant_forecaster/
├── data/
│   ├── historical_covers.csv      ← Your primary dataset (template included)
│   ├── ingredients.csv            ← Ingredient catalogue with costs/shelf life
│   ├── staff_roles.csv            ← Role definitions and ratios
│   └── feedback_log.csv           ← Auto-updated by feedback API
├── models/
│   ├── forecaster.py              ← Core ML: CoverPredictor, StaffScheduler, IngredientOrderer
│   └── cover_predictor.pkl        ← Saved model (generated after training)
├── api/
│   └── app.py                     ← Flask REST API
├── templates/
│   └── index.html                 ← Dashboard HTML
├── static/
│   ├── css/style.css
│   └── js/app.js
├── scripts/
│   ├── generate_data.py           ← Generate 2yr synthetic dataset
│   ├── train.py                   ← Train the base model
│   └── demo.py                    ← CLI demo of the full pipeline
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data (or use your own CSV)
```bash
python scripts/generate_data.py
```
This creates `data/historical_covers.csv` with 2 years of realistic data.

**To use your own data**, create a CSV matching this schema:
```
date,day_of_week,hour,covers,weather,is_holiday,is_special_event,notes
2024-01-05,Friday,19,118,sunny,False,False,
2024-01-05,Friday,20,125,sunny,False,False,
```
- `date`: YYYY-MM-DD
- `hour`: integer 10–23
- `covers`: integer (customers served that hour)
- `weather`: sunny / cloudy / rainy / stormy
- `is_holiday`, `is_special_event`: True/False

### 3. Train the model
```bash
python scripts/train.py
```
Outputs cross-validated MAE and feature importances.

### 4. Run the CLI demo
```bash
python scripts/demo.py
```
Shows covers, staff, and ingredients in the terminal, then simulates a feedback correction.

### 5. Launch the web dashboard
```bash
python api/app.py
```
Open http://localhost:5000

---

## REST API Reference

### `POST /api/predict/covers`
```json
{ "date": "2024-03-20", "weather": "sunny", "is_holiday": false, "is_special_event": false }
```
Returns hourly cover predictions with confidence bounds.

### `POST /api/predict/staff`
Same body → returns role-by-role staffing schedule and daily cost.

### `POST /api/predict/ingredients`
```json
{ "start_date": "2024-03-20", "days_ahead": 5, "current_stock": {"chicken_breast": 3.0} }
```
Returns prioritised order list with urgency flags.

### `POST /api/predict/full`
Single call returning covers + staff + ingredients for a day.

### `POST /api/feedback` ← **The learning loop**
```json
{
  "date": "2024-03-20",
  "predicted_covers": 120,
  "actual_covers": 85,
  "weather": "rainy",
  "note": "Unexpected downpour all evening"
}
```
Updates the correction model immediately. Call this after every service.

### `GET /api/feedback/history`
Returns all submitted corrections and accuracy stats.

### `GET /api/model/status`
Returns CV MAE, feature importances, feedback stats, convergence metrics.

---

## Data Templates

### `data/historical_covers.csv`
Minimum required fields: `date, day_of_week, hour, covers, weather, is_holiday, is_special_event`

### `data/ingredients.csv`
Fields: `ingredient_name, unit, cost_per_unit, shelf_life_days, supplier_lead_days, min_order_qty, category, usage_per_cover`

### `data/staff_roles.csv`
Fields: `role, station, hourly_rate, covers_per_staff_hour, min_staff, max_staff`

---

## Retraining Schedule

| Trigger | Action |
|---------|--------|
| Weekly | `python scripts/train.py --retrain` with accumulated data |
| After 20+ feedback events | Recheck correction model bias |
| Major change (new menu, expansion) | Retrain + reset correction model |

---

## Extending the System

- **Add delivery/takeaway** — add a `channel` column and feature
- **Menu-level predictions** — extend `usage_per_cover` with dish-mix ratios
- **Supplier API integration** — wire `IngredientOrderer` to auto-submit purchase orders
- **POS integration** — auto-ingest actuals from your till system into feedback loop
- **Weather API** — pull real forecasts instead of manual weather input
