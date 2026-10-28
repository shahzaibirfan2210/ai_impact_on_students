import gzip
import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
TARGET = "Burnout_Risk_Level"
MODEL_DIR = ROOT / "models"
VISUAL_DIR = ROOT / "output" / "visual"
PUBLIC_DIR = ROOT / "public"


def load_dataset(dataset_path: Path | None) -> pd.DataFrame:
    if dataset_path is not None:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        return pd.read_csv(dataset_path)

    import kagglehub

    downloaded_path = Path(
        kagglehub.dataset_download("laveshjadon/ai-impact-on-students")
    )
    csv_files = list(downloaded_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in the downloaded dataset: {downloaded_path}"
        )
    preferred = downloaded_path / "ai_student_impact_dataset (1).csv"
    return pd.read_csv(preferred if preferred.is_file() else csv_files[0])


def input_schema(features: pd.DataFrame) -> list[dict]:
    schema = []
    for column in features.columns:
        series = features[column]
        if pd.api.types.is_bool_dtype(series.dtype):
            schema.append(
                {
                    "name": column,
                    "kind": "boolean",
                    "default": bool(series.mode().iloc[0]),
                }
            )
        elif pd.api.types.is_numeric_dtype(series.dtype):
            schema.append(
                {
                    "name": column,
                    "kind": "number",
                    "default": float(series.median()),
                    "minimum": float(series.min()),
                    "maximum": float(series.max()),
                    "integer": pd.api.types.is_integer_dtype(series.dtype),
                }
            )
        else:
            values = sorted(series.dropna().astype(str).unique().tolist())
            if not values:
                raise ValueError(f"Categorical feature {column!r} has no values.")
            schema.append(
                {
                    "name": column,
                    "kind": "category",
                    "choices": values,
                    "default": str(series.mode().iloc[0]),
                }
            )
    return schema


def make_pipeline(
    classifier,
    categorical_columns: list[str],
    numeric_columns: list[str],
    dense: bool = False,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def export_browser_model(pipeline: Pipeline, features: pd.DataFrame, test_metrics: dict) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["categorical"]
    categorical_columns = preprocessor.transformers_[0][2]
    numeric_columns = preprocessor.transformers_[1][2]

    feature_sources = []
    for column, categories in zip(categorical_columns, encoder.categories_):
        feature_sources.extend(
            {"column": column, "equals": str(category)} for category in categories
        )
    feature_sources.extend({"column": column} for column in numeric_columns)

    classifier = pipeline.named_steps["classifier"]
    browser_payload = {
        "target_column": TARGET,
        "classes": [str(value) for value in classifier.classes_],
        "selected_model": "Random forest (128 trees, browser edition)",
        "test_metrics": test_metrics,
        "features": input_schema(features),
        "feature_sources": feature_sources,
        "trees": [
            {
                "left": estimator.tree_.children_left.tolist(),
                "right": estimator.tree_.children_right.tolist(),
                "feature": estimator.tree_.feature.tolist(),
                "threshold": estimator.tree_.threshold.tolist(),
                "values": estimator.tree_.value[:, 0, :].tolist(),
            }
            for estimator in classifier.estimators_
        ],
    }

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    with (PUBLIC_DIR / "burnout_model.json.gz").open("wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", compresslevel=9, mtime=0) as compressed:
            compressed.write(
                json.dumps(browser_payload, separators=(",", ":")).encode("utf-8")
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select, evaluate, and save the student burnout classifier."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to the dataset CSV. Defaults to downloading it from Kaggle.",
    )
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain the target column {TARGET!r}.")
    data = data.dropna(subset=[TARGET]).copy()
    labels = data.pop(TARGET)
    features = data.drop(columns=["Student_ID"], errors="ignore")
    if features.empty or labels.nunique() < 2:
        raise ValueError("The dataset must contain features and at least two target classes.")

    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    if len(categorical_columns) + len(numeric_columns) != len(features.columns):
        raise ValueError("Every model feature must be categorical, numeric, or boolean.")

    train_validation_x, test_x, train_validation_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_validation_x,
        train_validation_y,
        test_size=0.25,
        random_state=42,
        stratify=train_validation_y,
    )

    candidates = [
        (
            "Decision tree baseline",
            DecisionTreeClassifier(max_depth=3, random_state=42),
            False,
        ),
        (
            "Random forest",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
            False,
        ),
        (
            "Tuned random forest",
            RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
            False,
        ),
        (
            "Browser random forest",
            RandomForestClassifier(
                n_estimators=128,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
            False,
        ),
        (
            "Histogram gradient boosting",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1,
                random_state=42,
            ),
            True,
        ),
    ]

    validation_results = []
    for name, classifier, dense in candidates:
        pipeline = make_pipeline(
            classifier, categorical_columns, numeric_columns, dense=dense
        )
        pipeline.fit(train_x, train_y)
        predictions = pipeline.predict(validation_x)
        validation_results.append(
            {
                "model": name,
                "accuracy": accuracy_score(validation_y, predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    validation_y, predictions
                ),
                "macro_f1": f1_score(validation_y, predictions, average="macro"),
                "classifier": classifier,
                "dense": dense,
            }
        )

    selected = max(validation_results, key=lambda result: result["accuracy"])
    final_pipeline = make_pipeline(
        selected["classifier"],
        categorical_columns,
        numeric_columns,
        dense=selected["dense"],
    )
    final_pipeline.fit(train_validation_x, train_validation_y)
    test_predictions = final_pipeline.predict(test_x)
    test_metrics = {
        "accuracy": accuracy_score(test_y, test_predictions),
        "balanced_accuracy": balanced_accuracy_score(test_y, test_predictions),
        "macro_f1": f1_score(test_y, test_predictions, average="macro"),
        "classification_report": classification_report(
            test_y,
            test_predictions,
            output_dict=True,
            zero_division=0,
        ),
        "test_rows": int(len(test_y)),
        "dataset_rows": int(len(data)),
        "train_rows": int(len(train_validation_y)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    browser_candidate = next(
        result
        for result in validation_results
        if result["model"] == "Browser random forest"
    )
    browser_pipeline = make_pipeline(
        browser_candidate["classifier"],
        categorical_columns,
        numeric_columns,
        dense=browser_candidate["dense"],
    )
    browser_pipeline.fit(train_validation_x, train_validation_y)
    browser_test_predictions = browser_pipeline.predict(test_x)
    browser_test_metrics = {
        "accuracy": accuracy_score(test_y, browser_test_predictions),
        "balanced_accuracy": balanced_accuracy_score(
            test_y, browser_test_predictions
        ),
        "macro_f1": f1_score(test_y, browser_test_predictions, average="macro"),
        "test_rows": int(len(test_y)),
    }
    export_browser_model(browser_pipeline, features, browser_test_metrics)

    validation_table = pd.DataFrame(
        [
            {key: result[key] for key in ("model", "accuracy", "balanced_accuracy", "macro_f1")}
            for result in validation_results
        ]
    ).sort_values("accuracy", ascending=False)
    validation_table.to_csv(MODEL_DIR / "validation_comparison.csv", index=False)
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "selected_model": selected["model"],
                "validation_accuracy": selected["accuracy"],
                "test": test_metrics,
                "browser_model_validation_accuracy": browser_candidate["accuracy"],
                "browser_model_test": browser_test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = {
        "pipeline": final_pipeline,
        "feature_columns": features.columns.tolist(),
        "input_schema": input_schema(features),
        "target_column": TARGET,
        "selected_model": selected["model"],
        "validation_accuracy": selected["accuracy"],
        "test_metrics": test_metrics,
    }
    joblib.dump(bundle, MODEL_DIR / "burnout_model.joblib", compress=3)

    classes = final_pipeline.classes_
    matrix = confusion_matrix(test_y, test_predictions, labels=classes)
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axis,
    )
    axis.set_title(f"Confusion Matrix (Test Accuracy: {test_metrics['accuracy']:.1%})")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(VISUAL_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    classifier = final_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        feature_names = final_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        importances = pd.Series(
            classifier.feature_importances_, index=feature_names
        ).nlargest(15).sort_values()
        fig, axis = plt.subplots(figsize=(10, 6))
        importances.plot(kind="barh", ax=axis, color="teal")
        axis.set_title("Top 15 Feature Importances")
        axis.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(VISUAL_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)

    print(f"Dataset rows: {len(data):,}")
    print("Validation leaderboard (model selection used validation accuracy):")
    print(validation_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected model: {selected['model']}")
    print(f"Untouched test accuracy: {test_metrics['accuracy']:.1%}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.1%}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.3f}")
    print(f"Saved model: {MODEL_DIR / 'burnout_model.joblib'}")
    print(
        f"Browser model test accuracy: {browser_test_metrics['accuracy']:.1%} "
        f"(saved as {PUBLIC_DIR / 'burnout_model.json.gz'})"
    )


if __name__ == "__main__":
    main()
