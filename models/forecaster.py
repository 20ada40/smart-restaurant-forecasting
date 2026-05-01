"""
Restaurant Forecasting System - Core Prediction Engine
Uses gradient boosting for predictions + online learning via SGD for corrections.
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class CoverPrediction:
    date: str
    hour: int
    predicted_covers: float
    lower_bound: float
    upper_bound: float
    confidence: float
    factors: dict


@dataclass
class StaffRequirement:
    role: str
    station: str
    hour: int
    count: int
    hourly_rate: float
    total_cost: float


@dataclass
class IngredientOrder:
    ingredient: str
    unit: str
    quantity_to_order: float
    current_stock_estimate: float
    shelf_life_days: int
    lead_days: int
    cost: float
    urgency: str  # "critical", "normal", "buffer"


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────

WEATHER_CODES = {"sunny": 1.0, "cloudy": 0.85, "rainy": 0.65, "stormy": 0.45}

DAY_WEIGHTS = {
    "Monday": 0.70, "Tuesday": 0.72, "Wednesday": 0.78,
    "Thursday": 0.82, "Friday": 1.00, "Saturday": 1.15, "Sunday": 1.05
}

HOUR_CURVES = {
    # Relative traffic multiplier by hour
    10: 0.20, 11: 0.45, 12: 0.85, 13: 1.00, 14: 0.75, 15: 0.40,
    16: 0.25, 17: 0.50, 18: 0.80, 19: 1.00, 20: 0.95, 21: 0.70,
    22: 0.40, 23: 0.20
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw data into ML features."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week_num"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week_num"].isin([5, 6]).astype(int)
    df["day_weight"] = df["day_of_week"].map(DAY_WEIGHTS).fillna(0.75)
    df["hour_curve"] = df["hour"].map(HOUR_CURVES).fillna(0.3)
    df["weather_score"] = df["weather"].map(WEATHER_CODES).fillna(0.80)
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_special_event"] = df["is_special_event"].astype(int)

    # Interaction features
    df["weekend_x_weather"] = df["is_weekend"] * df["weather_score"]
    df["event_x_hour"] = df["is_special_event"] * df["hour_curve"]
    df["day_x_hour"] = df["day_weight"] * df["hour_curve"]

    # Seasonal sine/cosine encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


FEATURE_COLS = [
    "day_of_week_num", "month", "week_of_year", "is_weekend",
    "day_weight", "hour_curve", "weather_score", "is_holiday",
    "is_special_event", "weekend_x_weather", "event_x_hour",
    "day_x_hour", "month_sin", "month_cos", "hour_sin", "hour_cos", "hour"
]


# ─────────────────────────────────────────────
# Cover Predictor (with online learning)
# ─────────────────────────────────────────────

class CoverPredictor:
    """
    Two-layer prediction:
    1. GradientBoosting trained on historical data (base model)
    2. SGDRegressor that learns correction coefficients from manager feedback
    The correction layer uses residuals: actual - base_prediction
    """

    def __init__(self):
        self.base_model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr", GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            ))
        ])
        # Online correction model — learns from feedback
        self.correction_model = SGDRegressor(
            learning_rate="adaptive", eta0=0.01,
            max_iter=1000, random_state=42
        )
        self.correction_scaler = StandardScaler()
        self.correction_fitted = False
        self.is_trained = False
        self.cv_score = None
        self.feature_importance = {}
        self.feedback_history = []  # List of (features, residual)

    def train(self, df: pd.DataFrame):
        logger.info("Training base cover model...")
        df_feat = build_features(df)
        X = df_feat[FEATURE_COLS].values
        y = df_feat["covers"].values

        scores = cross_val_score(self.base_model, X, y, cv=5,
                                 scoring="neg_mean_absolute_error")
        self.cv_score = -scores.mean()
        logger.info(f"Base model CV MAE: {self.cv_score:.2f} covers")

        self.base_model.fit(X, y)
        self.is_trained = True

        # Store feature importance
        gbr = self.base_model.named_steps["gbr"]
        self.feature_importance = dict(zip(FEATURE_COLS, gbr.feature_importances_))

        # Warm-start correction model with zeros
        self._init_correction_model(X)
        self._save()

    def _init_correction_model(self, X_sample):
        X_scaled = self.correction_scaler.fit_transform(X_sample)
        residuals = np.zeros(len(X_sample))
        self.correction_model.partial_fit(X_scaled, residuals)
        self.correction_fitted = True

    def predict(self, date_str: str, hour: int, weather: str = "sunny",
                is_holiday: bool = False, is_special_event: bool = False) -> CoverPrediction:
        dt = pd.to_datetime(date_str)
        row = pd.DataFrame([{
            "date": date_str,
            "day_of_week": dt.strftime("%A"),
            "hour": hour,
            "weather": weather,
            "is_holiday": is_holiday,
            "is_special_event": is_special_event,
        }])
        row_feat = build_features(row)
        X = row_feat[FEATURE_COLS].values

        base_pred = float(self.base_model.predict(X)[0])

        correction = 0.0
        if self.correction_fitted:
            X_scaled = self.correction_scaler.transform(X)
            correction = float(self.correction_model.predict(X_scaled)[0])

        raw_pred = max(0, base_pred + correction)

        # Uncertainty bounds (±15% base, shrinks with more feedback)
        uncertainty = max(0.08, 0.15 - len(self.feedback_history) * 0.002)
        lower = max(0, raw_pred * (1 - uncertainty))
        upper = raw_pred * (1 + uncertainty)

        # Confidence score: improves with feedback
        confidence = min(0.95, 0.60 + len(self.feedback_history) * 0.01)

        return CoverPrediction(
            date=date_str,
            hour=hour,
            predicted_covers=round(raw_pred, 1),
            lower_bound=round(lower, 1),
            upper_bound=round(upper, 1),
            confidence=round(confidence, 2),
            factors={
                "base_prediction": round(base_pred, 1),
                "correction": round(correction, 1),
                "weather": weather,
                "day_weight": DAY_WEIGHTS.get(dt.strftime("%A"), 0.75),
                "hour_curve": HOUR_CURVES.get(hour, 0.3),
                "is_holiday": is_holiday,
                "is_special_event": is_special_event,
            }
        )

    def apply_feedback(self, date_str: str, hour: int, weather: str,
                       predicted: float, actual: float,
                       is_holiday: bool = False, is_special_event: bool = False,
                       note: str = ""):
        """
        Online learning: feed residual back into correction model.
        Uses exponential weighting — recent corrections matter more.
        """
        dt = pd.to_datetime(date_str)
        row = pd.DataFrame([{
            "date": date_str,
            "day_of_week": dt.strftime("%A"),
            "hour": hour,
            "weather": weather,
            "is_holiday": is_holiday,
            "is_special_event": is_special_event,
        }])
        row_feat = build_features(row)
        X = row_feat[FEATURE_COLS].values
        residual = actual - predicted

        X_scaled = self.correction_scaler.transform(X)
        self.correction_model.partial_fit(X_scaled, [residual])
        self.correction_fitted = True

        self.feedback_history.append({
            "date": date_str, "hour": hour, "predicted": predicted,
            "actual": actual, "residual": residual, "note": note
        })
        logger.info(f"Feedback applied: predicted={predicted}, actual={actual}, "
                    f"residual={residual:+.1f}, note='{note}'")
        self._save()

    def get_feedback_stats(self) -> dict:
        if not self.feedback_history:
            return {"count": 0}
        residuals = [f["residual"] for f in self.feedback_history]
        return {
            "count": len(residuals),
            "mean_error": round(np.mean(residuals), 2),
            "mae": round(np.mean(np.abs(residuals)), 2),
            "std": round(np.std(residuals), 2),
            "recent_bias": round(np.mean(residuals[-5:]), 2) if len(residuals) >= 5 else None
        }

    def _save(self):
        with open(MODELS_DIR / "cover_predictor.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls) -> "CoverPredictor":
        path = MODELS_DIR / "cover_predictor.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError("Model not trained yet. Run: python scripts/train.py")


# ─────────────────────────────────────────────
# Staff Scheduler
# ─────────────────────────────────────────────

class StaffScheduler:
    """Determines staffing levels from predicted covers per hour."""

    def __init__(self):
        self.roles_df = pd.read_csv(DATA_DIR / "staff_roles.csv")

    def schedule(self, hourly_covers: dict[int, float]) -> list[StaffRequirement]:
        """
        hourly_covers: {hour: predicted_covers}
        Returns list of StaffRequirement per role per hour.
        """
        requirements = []
        for hour, covers in hourly_covers.items():
            for _, role in self.roles_df.iterrows():
                if role["role"] == "manager":
                    count = role["min_staff"]
                elif role["role"] == "head_chef":
                    count = 1 if covers > 0 else 0
                elif role["role"] == "sous_chef":
                    count = 1 if covers > 60 else 0
                elif role["role"] == "pastry_chef":
                    count = 1 if covers > 80 else 0
                elif role["role"] in ("prep_cook",) and hour > 17:
                    count = 0  # Prep cooks work mornings
                else:
                    raw = covers / role["covers_per_staff_hour"]
                    count = int(np.ceil(raw))

                count = max(int(role["min_staff"]), min(count, int(role["max_staff"])))

                requirements.append(StaffRequirement(
                    role=role["role"],
                    station=role["station"],
                    hour=hour,
                    count=count,
                    hourly_rate=role["hourly_rate"],
                    total_cost=round(count * role["hourly_rate"], 2)
                ))

        return requirements

    def daily_summary(self, requirements: list[StaffRequirement]) -> dict:
        """Aggregate staff schedule into daily summary."""
        by_role: dict[str, dict] = {}
        for r in requirements:
            if r.role not in by_role:
                by_role[r.role] = {
                    "role": r.role, "station": r.station,
                    "peak_count": 0, "total_hours": 0, "total_cost": 0.0,
                    "hourly_rate": r.hourly_rate
                }
            entry = by_role[r.role]
            entry["peak_count"] = max(entry["peak_count"], r.count)
            entry["total_hours"] += r.count
            entry["total_cost"] += r.total_cost

        total_cost = sum(v["total_cost"] for v in by_role.values())
        return {
            "by_role": list(by_role.values()),
            "total_labor_cost": round(total_cost, 2),
            "total_staff_hours": sum(v["total_hours"] for v in by_role.values())
        }


# ─────────────────────────────────────────────
# Ingredient Orderer
# ─────────────────────────────────────────────

class IngredientOrderer:
    """
    Calculates ingredient orders accounting for:
    - Expected usage from predicted covers
    - Shelf life (avoid over-ordering perishables)
    - Supplier lead times (order ahead)
    - Safety buffer stock
    """
    SAFETY_BUFFER = 1.15  # 15% buffer for uncertainty

    def __init__(self):
        self.ingredients_df = pd.read_csv(DATA_DIR / "ingredients.csv")

    def calculate_orders(self, daily_cover_totals: dict[str, float],
                         current_stock: Optional[dict] = None,
                         order_date: Optional[str] = None) -> list[IngredientOrder]:
        """
        daily_cover_totals: {"2024-03-20": 180, "2024-03-21": 95, ...}
        current_stock: {ingredient_name: qty_on_hand} (optional)
        order_date: date we're placing the order (default: today)
        """
        if order_date is None:
            order_date = date.today()
        else:
            order_date = pd.to_datetime(order_date).date()

        current_stock = current_stock or {}
        orders = []

        for _, ing in self.ingredients_df.iterrows():
            name = ing["ingredient_name"]
            usage_per_cover = ing["usage_per_cover"]
            shelf_life = int(ing["shelf_life_days"])
            lead_days = int(ing["supplier_lead_days"])

            # Covers we need to serve, accounting for lead time
            relevant_days = sorted([
                (d, c) for d, c in daily_cover_totals.items()
                if pd.to_datetime(d).date() >= order_date + timedelta(days=lead_days)
                and pd.to_datetime(d).date() < order_date + timedelta(days=lead_days + shelf_life)
            ])

            total_needed = sum(c * usage_per_cover for _, c in relevant_days)
            total_needed *= self.SAFETY_BUFFER

            on_hand = current_stock.get(name, 0.0)
            # Only order what we don't have; round up to min order qty
            qty_to_order = max(0, total_needed - on_hand)
            min_order = ing["min_order_qty"]
            if qty_to_order > 0:
                qty_to_order = max(qty_to_order, min_order)
                # Round up to nearest min_order multiple
                qty_to_order = np.ceil(qty_to_order / min_order) * min_order

            # Urgency classification
            if lead_days >= shelf_life:
                urgency = "critical"
            elif shelf_life <= 3:
                urgency = "critical"
            elif shelf_life <= 7:
                urgency = "normal"
            else:
                urgency = "buffer"

            orders.append(IngredientOrder(
                ingredient=name,
                unit=ing["unit"],
                quantity_to_order=round(qty_to_order, 2),
                current_stock_estimate=on_hand,
                shelf_life_days=shelf_life,
                lead_days=lead_days,
                cost=round(qty_to_order * ing["cost_per_unit"], 2),
                urgency=urgency
            ))

        # Sort: critical first, then by cost desc
        urgency_order = {"critical": 0, "normal": 1, "buffer": 2}
        orders.sort(key=lambda x: (urgency_order[x.urgency], -x.cost))
        return orders

    def order_cost_summary(self, orders: list[IngredientOrder]) -> dict:
        total = sum(o.cost for o in orders)
        by_cat = {}
        ing_map = self.ingredients_df.set_index("ingredient_name")["category"].to_dict()
        for o in orders:
            cat = ing_map.get(o.ingredient, "other")
            by_cat[cat] = by_cat.get(cat, 0.0) + o.cost
        return {
            "total_food_cost": round(total, 2),
            "by_category": {k: round(v, 2) for k, v in sorted(by_cat.items())}
        }
