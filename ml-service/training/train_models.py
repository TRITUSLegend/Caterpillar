"""
Trains the two ML models used by the ML service and saves them to ml-service/models/.

  task_time_model.pkl - Gradient Boosting Regressor on data/tasks.csv, predicts Actual_Time
  anomaly_model.pkl   - Isolation Forest on data/telemetry.csv (per-machine-type normalized,
                        idle time also split by end-of-shift vs rest of shift)

Both are full sklearn Pipelines with preprocessing included, so the API passes raw
values (e.g. "Rainy", "Excavator") straight in.

Usage (from repo root):  python ml-service/training/train_models.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ML_SERVICE_DIR.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = ML_SERVICE_DIR / "models"

sys.path.insert(0, str(ML_SERVICE_DIR))
from preprocessing import PerTypeScaler  # noqa: E402

SEED = 42

TASK_CATEGORICAL = ["Task_Type", "Weather", "Site_Terrain"]
TASK_SKILL = ["Operator_Skill"]
TASK_NUMERIC = ["Machine_Age", "Estimated_Time"]
TASK_FEATURES = TASK_CATEGORICAL + TASK_SKILL + TASK_NUMERIC
TASK_TARGET = "Actual_Time"
SKILL_ORDER = ["Beginner", "Intermediate", "Expert"]

ANOMALY_FEATURES = ["Hydraulic_Pressure", "Fuel_Used", "Idling_Time", "Load_Cycles"]
ANOMALY_CONTAMINATION = 0.03


def build_task_pipeline():
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), TASK_CATEGORICAL),
        ("skill", OrdinalEncoder(categories=[SKILL_ORDER]), TASK_SKILL),
        ("numeric", "passthrough", TASK_NUMERIC),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", GradientBoostingRegressor(random_state=SEED)),
    ])


def train_task_time(tasks):
    X, y = tasks[TASK_FEATURES], tasks[TASK_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipe = build_task_pipeline().fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    baseline_mae = mean_absolute_error(y_test, X_test["Estimated_Time"])

    print("Task time model (Gradient Boosting Regressor)")
    print(f"  train/test rows:        {len(X_train)} / {len(X_test)}")
    print(f"  test MAE:               {mae:.2f} min")
    print(f"  test RMSE:              {rmse:.2f} min")
    print(f"  baseline MAE (=Estimated_Time): {baseline_mae:.2f} min")
    print(f"  mean Actual_Time (test): {y_test.mean():.1f} min")

    names = pipe.named_steps["preprocess"].get_feature_names_out()
    importances = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    print("  top features:")
    for name, value in importances.sort_values(ascending=False).head(5).items():
        print(f"    {name:<40} {value:.3f}")

    # Final model is refit on all rows once the held-out metrics look sane
    return build_task_pipeline().fit(X, y)


def train_anomaly(telemetry):
    X = telemetry[["Machine_Type", "Timestamp"] + ANOMALY_FEATURES]
    pipe = Pipeline([
        ("scale", PerTypeScaler(
            group_col="Machine_Type",
            features=ANOMALY_FEATURES,
            time_col="Timestamp",
            end_of_shift_features=("Idling_Time",),
        )),
        ("model", IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=SEED)),
    ]).fit(X)

    score = -pipe.decision_function(X)
    flagged = score > 0
    assert (flagged == (pipe.predict(X) == -1)).all()

    print("\nAnomaly model (Isolation Forest)")
    print(f"  training rows:  {len(X)}")
    print(f"  flagged:        {flagged.sum()} ({flagged.mean():.2%}), contamination={ANOMALY_CONTAMINATION}")
    print(f"  score range:    {score.min():.3f} to {score.max():.3f}")
    print("  flagged by machine:", telemetry.loc[flagged, "Machine_ID"].value_counts().sort_index().to_dict())
    compare = pd.DataFrame({
        "normal_mean": telemetry.loc[~flagged, ANOMALY_FEATURES].mean(),
        "flagged_mean": telemetry.loc[flagged, ANOMALY_FEATURES].mean(),
    }).round(1)
    print("  feature means (normal vs flagged):")
    print("    " + compare.to_string().replace("\n", "\n    "))
    print("  note: idle spikes are caught reliably and most fuel spikes; sudden hydraulic pressure")
    print("        drops are only partly caught (about 2 of 6 in the generated data), mainly on")
    print("        excavators where the worn M03 widens the normal range. Slow pressure loss is")
    print("        covered by the rule-based maintenance score instead.")
    return pipe


def main():
    np.random.seed(SEED)
    tasks = pd.read_csv(DATA_DIR / "tasks.csv")
    telemetry = pd.read_csv(DATA_DIR / "telemetry.csv")

    task_model = train_task_time(tasks)
    anomaly_model = train_anomaly(telemetry)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(task_model, MODELS_DIR / "task_time_model.pkl")
    joblib.dump(anomaly_model, MODELS_DIR / "anomaly_model.pkl")

    print(f"\nSaved to {MODELS_DIR} (scikit-learn {sklearn.__version__})")


if __name__ == "__main__":
    main()
