"""
FlightInsight — Trening XGBoost modela.

Procesna veriga:
  1. Naloži flights.csv
  2. Pripravi X, y (features + target)
  3. Train/test split
  4. Sklearn Pipeline: ColumnTransformer + XGBRegressor
  5. Trening z early stopping
  6. Evalvacija (MAE, RMSE, R²)
  7. XGBoost feature importance (gain, weight, cover)
  8. MLflow tracking (parametri, metrike, artefakti)
  9. Shrani model + pipeline

MLflow tracking gre na DagsHub (https://dagshub.com/<user>/<repo>.mlflow).
Za auth potrebuješ env vars MLFLOW_TRACKING_USERNAME in MLFLOW_TRACKING_PASSWORD.
"""

import os
import sys
import warnings

import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")  # ne odpiraj GUI
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import build_preprocessor, select_features, get_feature_names

warnings.filterwarnings("ignore")

DATA_PATH = "data/preprocessed/flights.csv"
MODEL_DIR = "models"
REPORTS_DIR = "reports"

DAGSHUB_USER = os.environ.get("DAGSHUB_USER", "MitrovicEmilija")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO", "Flight-Insight")
MLFLOW_URI = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"


def setup_mlflow() -> bool:
    """Konfiguriraj MLflow tracking. Vrne True če je auth nastavljen."""
    if not os.environ.get("MLFLOW_TRACKING_USERNAME"):
        print("OPOZORILO: MLFLOW_TRACKING_USERNAME ni nastavljen!")
        print("  Nastavi: export MLFLOW_TRACKING_USERNAME=<dagshub_user>")
        print("  Nastavi: export MLFLOW_TRACKING_PASSWORD=<dagshub_token>")
        print("  Pipeline bo nadaljeval brez MLflow tracking-a.")
        return False

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("flight_delay_prediction")
    print(f"MLflow tracking URI: {MLFLOW_URI}")
    return True


def load_and_split(params: dict) -> tuple:
    """Naloži podatke in razdeli na train/test."""
    print(f"Nalagam {DATA_PATH}...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Vrstic: {len(df):,}")

    # Filtriraj na 2024 (model ne sme videti 2025!)
    if "Year" in df.columns:
        n_before = len(df)
        df = df[df["Year"] == 2024].copy()
        print(f"  Filtrirano na 2024: {len(df):,} (odstranjenih {n_before - len(df):,})")

    # Vzorec za RAM
    sample_size = params.get("sample_size", None)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=params["random_state"]).reset_index(drop=True)
        print(f"  Vzorec za trening: {len(df):,}")

    # Features in target
    X, y = select_features(df)
    print(f"  Features: {X.shape[1]}, Target: {y.name}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, params: dict) -> Pipeline:
    """Zgradi in nauči sklearn Pipeline (preprocessor + XGBoost)."""
    print("\nGradnja pipeline-a...")
    preprocessor = build_preprocessor()

    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        random_state=params["random_state"],
        tree_method="hist",
        early_stopping_rounds=20,
        eval_metric="mae",
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("xgboost", model),
    ])

    print("Učenje modela...")
    # Za early stopping potrebujemo eval_set, zato preprocesiramo ročno
    X_train_processed = preprocessor.fit_transform(X_train)

    # Mali validation set za early stopping
    n_val = min(50_000, len(X_train) // 10)
    X_train_p, X_val_p = X_train_processed[:-n_val], X_train_processed[-n_val:]
    y_train_p, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    model.fit(
        X_train_p, y_train_p,
        eval_set=[(X_val_p, y_val)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best score (MAE): {model.best_score:.4f}")

    return pipeline


def evaluate(pipeline: Pipeline, X_test, y_test) -> tuple[dict, np.ndarray]:
    """Evalviraj model in vrni metrike."""
    print("\nEvalvacija na test setu...")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["xgboost"]

    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)

    metrics = {
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_mse": float(mean_squared_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
    }

    print(f"  MAE:  {metrics['test_mae']:.2f} min")
    print(f"  RMSE: {metrics['test_rmse']:.2f} min")
    print(f"  R²:   {metrics['test_r2']:.4f}")

    return metrics, y_pred


def feature_importance_analysis(pipeline: Pipeline, X_sample: pd.DataFrame) -> tuple[str, str]:
    """
    XGBoost native feature importance analiza.
    Generira 2 grafa:
      - bar plot top 20 features (po gain)
      - primerjava gain vs weight vs cover
    """
    print("\nFeature importance analiza...")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["xgboost"]

    # Pridobi feature names po preprocessing-u
    feature_names = get_feature_names(preprocessor, X_sample)

    # XGBoost ima 3 vrste importance:
    # - gain: povprečni dobiček (informacijski) ko je feature uporabljen v razdelitvi
    # - weight: kolikokrat je feature uporabljen v drevesih
    # - cover: povprečno število primerov ki gredo skozi razdelitve s tem feature-jem
    importance_types = ["gain", "weight", "cover"]
    importances = {}

    booster = model.get_booster()
    for imp_type in importance_types:
        imp_dict = booster.get_score(importance_type=imp_type)
        # Map f0, f1, ... -> feature names
        importance_array = np.zeros(len(feature_names))
        for f_key, f_val in imp_dict.items():
            idx = int(f_key.replace("f", ""))
            if idx < len(feature_names):
                importance_array[idx] = f_val
        importances[imp_type] = importance_array

    # ===== Graf 1: Top 20 features po GAIN =====
    gain_values = importances["gain"]
    top_idx = np.argsort(gain_values)[-20:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    top_values = gain_values[top_idx]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_values[::-1], color="steelblue")
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Importance (gain)")
    plt.title("Top 20 najpomembnejših faktorjev za napoved zamude\n(višji gain = pomembnejši)")
    plt.tight_layout()

    bar_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Shranjeno: {bar_path}")

    # ===== Graf 2: Primerjava gain vs weight vs cover =====
    # Top 15 po gain
    top15_idx = np.argsort(gain_values)[-15:][::-1]
    top15_features = [feature_names[i] for i in top15_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, imp_type in zip(axes, importance_types):
        values = importances[imp_type][top15_idx]
        # Normaliziraj na 0-1
        if values.max() > 0:
            values = values / values.max()
        ax.barh(range(len(top15_features)), values[::-1], color="steelblue")
        ax.set_yticks(range(len(top15_features)))
        ax.set_yticklabels(top15_features[::-1], fontsize=9)
        ax.set_xlabel(f"Normalized {imp_type}")
        ax.set_title(f"Importance: {imp_type}")
        ax.set_xlim(0, 1.05)

    plt.suptitle("Feature importance — primerjava 3 metrik", fontsize=14)
    plt.tight_layout()

    comparison_path = os.path.join(REPORTS_DIR, "feature_importance_comparison.png")
    plt.savefig(comparison_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Shranjeno: {comparison_path}")

    # ===== CSV s feature importance =====
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "gain": importances["gain"],
        "weight": importances["weight"],
        "cover": importances["cover"],
    }).sort_values("gain", ascending=False)

    csv_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"  Shranjeno: {csv_path}")

    # Izpiši top 10
    print("\n  Top 10 features po gain:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:30s} gain={row['gain']:.2f}")

    return bar_path, comparison_path


def save_model(pipeline: Pipeline, metrics: dict) -> dict:
    """Shrani model + metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nModel shranjen: {model_path}")

    metadata = {
        "model_type": "XGBRegressor",
        "metrics": metrics,
        "n_features": int(pipeline.named_steps["xgboost"].n_features_in_),
    }

    metadata_path = os.path.join(MODEL_DIR, "metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f)

    return {"model": model_path, "metadata": metadata_path}


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]

    # 1. MLflow setup
    use_mlflow = setup_mlflow()

    # 2. Naloži in razdeli podatke
    X_train, X_test, y_train, y_test = load_and_split(train_params)

    # 3. Trening v MLflow run kontekstu
    if use_mlflow:
        mlflow.start_run(run_name="xgboost_baseline")
        mlflow.log_params(train_params)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

    try:
        # Train
        pipeline = train_model(X_train, y_train, train_params)

        # Evaluate
        metrics, y_pred = evaluate(pipeline, X_test, y_test)

        # Feature importance
        bar_path, comparison_path = feature_importance_analysis(pipeline, X_test)

        # Save model
        artifacts = save_model(pipeline, metrics)

        # MLflow logging
        if use_mlflow:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_artifact(artifacts["model"])
            mlflow.log_artifact(artifacts["metadata"])
            mlflow.log_artifact(bar_path)
            mlflow.log_artifact(comparison_path)
            mlflow.log_artifact(os.path.join(REPORTS_DIR, "feature_importance.csv"))

            print("\n✓ MLflow run zabeležen")

    finally:
        if use_mlflow:
            mlflow.end_run()

    print("\n" + "=" * 60)
    print("Trening uspešno zaključen!")
    print("=" * 60)


if __name__ == "__main__":
    main()
