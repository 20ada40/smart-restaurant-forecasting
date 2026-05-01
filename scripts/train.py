"""
Train the cover prediction model on historical data.
Run once initially, then periodically as more data accumulates.

Usage:
    python scripts/train.py
    python scripts/train.py --retrain   # force retrain even if model exists
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.forecaster import CoverPredictor, MODELS_DIR


def train(force: bool = False):
    model_path = MODELS_DIR / "cover_predictor.pkl"
    if model_path.exists() and not force:
        print("Model already exists. Use --retrain to force retraining.")
        return

    data_path = Path(__file__).parent.parent / "data" / "historical_covers.csv"
    if not data_path.exists():
        print(f"ERROR: No data found at {data_path}")
        print("Please create the historical_covers.csv file first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} historical records from {data_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Covers range: {df['covers'].min()} - {df['covers'].max()}")

    predictor = CoverPredictor()
    predictor.train(df)

    print(f"\n✓ Model trained and saved to {MODELS_DIR}")
    print(f"  Cross-validated MAE: {predictor.cv_score:.2f} covers")
    print(f"\n  Top 5 important features:")
    sorted_features = sorted(predictor.feature_importance.items(),
                             key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:5]:
        print(f"    {feat:25s} {imp:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train restaurant forecasting model")
    parser.add_argument("--retrain", action="store_true", help="Force retrain")
    args = parser.parse_args()
    train(force=args.retrain)
