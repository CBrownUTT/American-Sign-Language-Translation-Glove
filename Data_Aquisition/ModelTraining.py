import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

RAW_CSV_FILE = "glove_sequence_data_total.csv"
OUTPUT_DIR = "training_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Must match the Arduino .ino header exactly
FEATURE_COLUMNS = [
    "hall_thumb", "hall_index", "hall_middle", "hall_ring", "hall_pinky",
    "imu_ax", "imu_ay", "imu_az",
    "imu_gx", "imu_gy", "imu_gz",
    "imu_ex", "imu_ey", "imu_ez",
    "contact_p", "contact_i", "contact_m", "contact_um"
]


# Put dynamic-sign labels here. Everything else is treated as static.
DYNAMIC_LABELS = {"J", "Z"}


def safe_interp(arr, positions):
    if len(arr) == 1:
        return [arr[0] for _ in positions]
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.array(positions)
    return np.interp(x_new, x_old, arr)


def sequence_to_features(seq_df):
    seq_df = seq_df.sort_values("frame_idx").reset_index(drop=True)
    t = seq_df["timestamp_ms"].to_numpy(dtype=float) / 1000.0

    if len(t) < 2:
        return None

    duration = max(t[-1] - t[0], 1e-6)
    feat = {
        "duration_s": duration,
        "num_frames": len(seq_df)
    }

    for col in FEATURE_COLUMNS:
        arr = seq_df[col].to_numpy(dtype=float)

        feat[f"{col}_first"] = arr[0]
        feat[f"{col}_last"] = arr[-1]
        feat[f"{col}_mean"] = np.mean(arr)
        feat[f"{col}_min"] = np.min(arr)
        feat[f"{col}_max"] = np.max(arr)
        feat[f"{col}_std"] = np.std(arr)
        feat[f"{col}_range"] = np.max(arr) - np.min(arr)
        feat[f"{col}_delta"] = arr[-1] - arr[0]
        feat[f"{col}_abs_change_sum"] = np.sum(np.abs(np.diff(arr)))
        feat[f"{col}_slope"] = (arr[-1] - arr[0]) / duration

        p0, p25, p50, p75, p100 = safe_interp(arr, [0.0, 0.25, 0.50, 0.75, 1.0])
        feat[f"{col}_p0"] = p0
        feat[f"{col}_p25"] = p25
        feat[f"{col}_p50"] = p50
        feat[f"{col}_p75"] = p75
        feat[f"{col}_p100"] = p100

    accel_mag = np.sqrt(
        seq_df["imu_ax"].to_numpy(dtype=float) ** 2 +
        seq_df["imu_ay"].to_numpy(dtype=float) ** 2 +
        seq_df["imu_az"].to_numpy(dtype=float) ** 2
    )
    gyro_mag = np.sqrt(
        seq_df["imu_gx"].to_numpy(dtype=float) ** 2 +
        seq_df["imu_gy"].to_numpy(dtype=float) ** 2 +
        seq_df["imu_gz"].to_numpy(dtype=float) ** 2
    )
    euler_mag = np.sqrt(
        seq_df["imu_ex"].to_numpy(dtype=float) ** 2 +
        seq_df["imu_ey"].to_numpy(dtype=float) ** 2 +
        seq_df["imu_ez"].to_numpy(dtype=float) ** 2
    )

    feat["accel_mag_mean"] = np.mean(accel_mag)
    feat["accel_mag_max"] = np.max(accel_mag)
    feat["accel_mag_std"] = np.std(accel_mag)
    feat["accel_mag_abs_change_sum"] = np.sum(np.abs(np.diff(accel_mag)))

    feat["gyro_mag_mean"] = np.mean(gyro_mag)
    feat["gyro_mag_max"] = np.max(gyro_mag)
    feat["gyro_mag_std"] = np.std(gyro_mag)
    feat["gyro_mag_abs_change_sum"] = np.sum(np.abs(np.diff(gyro_mag)))

    feat["euler_mag_mean"] = np.mean(euler_mag)
    feat["euler_mag_max"] = np.max(euler_mag)
    feat["euler_mag_std"] = np.std(euler_mag)
    feat["euler_mag_abs_change_sum"] = np.sum(np.abs(np.diff(euler_mag)))

    return feat


def build_sequence_feature_table(raw_df):
    rows = []

    for sequence_id, seq_df in raw_df.groupby("sequence_id"):
        label = seq_df["label"].iloc[0]
        feat = sequence_to_features(seq_df)
        if feat is None:
            continue
        feat["sequence_id"] = sequence_id
        feat["label"] = label
        rows.append(feat)

    return pd.DataFrame(rows)


def split_balanced_static_dynamic(feature_df, dynamic_labels=None, test_fraction=0.20, random_state=42):
    if feature_df.empty:
        raise ValueError("feature_df is empty.")

    if dynamic_labels is None:
        dynamic_labels = set()
    dynamic_labels = {str(x).upper() for x in dynamic_labels}

    sequence_meta = feature_df[["sequence_id", "label"]].drop_duplicates().copy()

    duplicate_seq = sequence_meta["sequence_id"].duplicated().any()
    if duplicate_seq:
        raise ValueError("A sequence_id appears with more than one label.")

    sequence_meta["group_type"] = sequence_meta["label"].apply(
        lambda x: "dynamic" if str(x).upper() in dynamic_labels else "static"
    )

    train_parts = []
    test_parts = []

    for group_type, group_df in sequence_meta.groupby("group_type"):
        label_counts = group_df["label"].value_counts().sort_index()
        if label_counts.empty:
            continue

        min_count = int(label_counts.min())
        if min_count < 2:
            raise ValueError(
                f"{group_type.title()} labels need at least 2 sequences each for a train/test split. "
                f"Smallest class has {min_count}."
            )

        per_label_test = max(1, int(round(min_count * test_fraction)))
        per_label_test = min(per_label_test, min_count - 1)

        for label, label_group in group_df.groupby("label"):
            label_group = label_group.sample(frac=1, random_state=random_state).reset_index(drop=True)

            test_seq_ids = set(label_group.iloc[:per_label_test]["sequence_id"])
            train_seq_ids = set(label_group.iloc[per_label_test:]["sequence_id"])

            label_feature_rows = feature_df[feature_df["sequence_id"].isin(set(label_group["sequence_id"]))].copy()

            test_rows = label_feature_rows[label_feature_rows["sequence_id"].isin(test_seq_ids)].copy()
            train_rows = label_feature_rows[label_feature_rows["sequence_id"].isin(train_seq_ids)].copy()

            if test_rows.empty or train_rows.empty:
                raise ValueError(
                    f"Split failed for label '{label}' in {group_type} group."
                )

            test_parts.append(test_rows)
            train_parts.append(train_rows)

    if not train_parts or not test_parts:
        raise ValueError("Split failed and produced no train/test partitions.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, test_df


def save_confusion_matrix(y_true, y_pred, labels, title, filename, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.show()
    plt.close(fig)


def save_feature_importance_plot(importances, top_n, title, filename):
    imp_df = importances.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.show()
    plt.close(fig)


def save_confidence_histogram(confidences, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=20)
    ax.set_title(title)
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.show()
    plt.close(fig)


def save_model_metrics_plot(metrics_df, filename="model_metrics_comparison.png"):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics_df["Model"]))
    width = 0.2

    ax.bar(x - 1.5 * width, metrics_df["Accuracy"], width, label="Accuracy")
    ax.bar(x - 0.5 * width, metrics_df["Precision"], width, label="Precision")
    ax.bar(x + 0.5 * width, metrics_df["Recall"], width, label="Recall")
    ax.bar(x + 1.5 * width, metrics_df["F1"], width, label="F1-score")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Model"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy, Precision, Recall, F1-score")
    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.show()
    plt.close(fig)


def print_split_summary(train_df, test_df):
    print("\nTrain sequences per class:")
    print(train_df["label"].value_counts().sort_index().to_string())

    print("\nTest sequences per class:")
    print(test_df["label"].value_counts().sort_index().to_string())


def print_group_split_summary(train_df, test_df, dynamic_labels=None):
    if dynamic_labels is None:
        dynamic_labels = set()
    dynamic_labels = {str(x).upper() for x in dynamic_labels}

    train_seq = train_df[["sequence_id", "label"]].drop_duplicates().copy()
    test_seq = test_df[["sequence_id", "label"]].drop_duplicates().copy()

    train_seq["group_type"] = train_seq["label"].apply(
        lambda x: "dynamic" if str(x).upper() in dynamic_labels else "static"
    )
    test_seq["group_type"] = test_seq["label"].apply(
        lambda x: "dynamic" if str(x).upper() in dynamic_labels else "static"
    )

    print("\nTrain sequences per class:")
    print(train_seq["label"].value_counts().sort_index().to_string())

    print("\nTest sequences per class:")
    print(test_seq["label"].value_counts().sort_index().to_string())

    print("\nTrain sequences per group:")
    print(train_seq["group_type"].value_counts().sort_index().to_string())

    print("\nTest sequences per group:")
    print(test_seq["group_type"].value_counts().sort_index().to_string())


def evaluate_model(name, model, X_train, y_train, X_test, y_test, class_labels):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        confidences = np.max(y_proba, axis=1)
        avg_confidence = float(np.mean(confidences))
        median_confidence = float(np.median(confidences))
    else:
        y_proba = None
        confidences = np.array([])
        avg_confidence = None
        median_confidence = None

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    print(cm_df.to_string())

    if avg_confidence is not None:
        print(f"\nAverage confidence: {avg_confidence:.4f}")
        print(f"Median confidence:  {median_confidence:.4f}")

    print("\nTraining details:")
    print("epochs: N/A (not used by decision trees / random forests)")
    print("batch_size: N/A (not used by decision trees / random forests)")
    print("iterations: N/A (not used in gradient-descent sense)")
    print(f"n_features_in_: {getattr(model, 'n_features_in_', 'N/A')}")

    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        print("\nTop 25 feature importances:")
        print(importances.head(25).to_string(index=False))
    else:
        importances = None

    pred_df = pd.DataFrame({
        "true": y_test.values,
        "pred": y_pred
    })

    wrong = pred_df[pred_df["true"] != pred_df["pred"]].copy()
    if not wrong.empty:
        pair_counts = wrong.groupby(["true", "pred"]).size().reset_index(name="count")
        pair_counts = pair_counts.sort_values("count", ascending=False)
        print("\nTop misclassification pairs:")
        print(pair_counts.head(10).to_string(index=False))
    else:
        print("\nNo misclassifications in the test set.")

    save_confusion_matrix(
        y_test, y_pred, class_labels,
        f"{name} Confusion Matrix",
        f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
        normalize=None
    )

    save_confusion_matrix(
        y_test, y_pred, class_labels,
        f"{name} Normalized Confusion Matrix",
        f"{name.lower().replace(' ', '_')}_confusion_matrix_normalized.png",
        normalize="true"
    )

    if importances is not None:
        save_feature_importance_plot(
            importances,
            top_n=20,
            title=f"{name} Top 20 Feature Importances",
            filename=f"{name.lower().replace(' ', '_')}_feature_importances.png"
        )

    if len(confidences) > 0:
        save_confidence_histogram(
            confidences,
            title=f"{name} Prediction Confidence Histogram",
            filename=f"{name.lower().replace(' ', '_')}_confidence_histogram.png"
        )

    return {
        "model": model,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confidences": confidences,
        "importances": importances
    }


def main():
    raw_df = pd.read_csv(RAW_CSV_FILE)

    print("CSV columns found:")
    print(sorted(raw_df.columns.tolist()))

    required = {"sequence_id", "label", "frame_idx", "timestamp_ms"} | set(FEATURE_COLUMNS)
    missing = required - set(raw_df.columns)
    if missing:
        print("\nExpected feature columns:")
        print(FEATURE_COLUMNS)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_df = build_sequence_feature_table(raw_df)

    if feature_df.empty:
        raise ValueError("No usable sequences found.")

    print("\nTotal sequences per class:")
    print(feature_df["label"].value_counts().sort_index().to_string())

    train_df, test_df = split_balanced_static_dynamic(
        feature_df,
        dynamic_labels=DYNAMIC_LABELS,
        test_fraction=0.20,
        random_state=42
    )

    print_group_split_summary(train_df, test_df, dynamic_labels=DYNAMIC_LABELS)

    X_train = train_df.drop(columns=["sequence_id", "label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["sequence_id", "label"])
    y_test = test_df["label"]

    class_labels = sorted(feature_df["label"].unique())

    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42
    )

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    dt_result = evaluate_model(
        "Decision Tree",
        decision_tree,
        X_train, y_train, X_test, y_test, class_labels
    )

    rf_result = evaluate_model(
        "Random Forest",
        random_forest,
        X_train, y_train, X_test, y_test, class_labels
    )

    metrics_df = pd.DataFrame([
        {
            "Model": "Decision Tree",
            "Accuracy": dt_result["accuracy"],
            "Precision": dt_result["precision"],
            "Recall": dt_result["recall"],
            "F1": dt_result["f1"]
        },
        {
            "Model": "Random Forest",
            "Accuracy": rf_result["accuracy"],
            "Precision": rf_result["precision"],
            "Recall": rf_result["recall"],
            "F1": rf_result["f1"]
        }
    ])

    print("\nModel comparison metrics:")
    print(metrics_df.to_string(index=False))

    save_model_metrics_plot(metrics_df)

    joblib.dump({
        "model": dt_result["model"],
        "feature_columns": list(X_train.columns),
        "sensor_columns": FEATURE_COLUMNS,
        "class_labels": class_labels
    }, os.path.join(OUTPUT_DIR, "asl_decision_tree.joblib"))

    joblib.dump({
        "model": rf_result["model"],
        "feature_columns": list(X_train.columns),
        "sensor_columns": FEATURE_COLUMNS,
        "class_labels": class_labels
    }, os.path.join(OUTPUT_DIR, "asl_random_forest.joblib"))

    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print("\nSaved files include:")
    print("- asl_decision_tree.joblib")
    print("- asl_random_forest.joblib")
    print("- decision_tree_confusion_matrix.png")
    print("- decision_tree_confusion_matrix_normalized.png")
    print("- decision_tree_feature_importances.png")
    print("- decision_tree_confidence_histogram.png")
    print("- random_forest_confusion_matrix.png")
    print("- random_forest_confusion_matrix_normalized.png")
    print("- random_forest_feature_importances.png")
    print("- random_forest_confidence_histogram.png")
    print("- model_metrics_comparison.png")


if __name__ == "__main__":
    main()
