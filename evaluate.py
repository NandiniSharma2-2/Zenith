"""
Zenith Sky Condition Monitoring — Comprehensive Model Evaluation & Comparison
Evaluates both the Baseline and V2 models on the held-out test datasets (unseen camera C009).
Calculates accuracy, precision, recall, F1, okta MAE, and generates confusion matrices.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Headless matplotlib configuration to avoid GUI blocking
os.environ.setdefault("KERAS_HOME", os.path.abspath(".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)

DATA_ROOT = Path("data/GOA-ALLSKY/ALLSKY-GOA")
REPORTS_DIR = Path("reports")
BASELINE_MODEL_PATH = Path("models/zenith_mobilenetv2.keras")
V2_MODEL_PATH = Path("models/zenith_v2.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = [str(i) for i in range(9)]


def load_test_dataset(subset: str = "night") -> Tuple[tf.data.Dataset, np.ndarray]:
    """Load held-out test dataset from unseen camera C009."""
    if subset not in {"day", "night"}:
        raise ValueError(f"Unknown test subset {subset!r}; expected 'day' or 'night'.")

    test_dir = DATA_ROOT / "test" / subset
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    if ds.class_names != CLASS_NAMES:
        raise ValueError(
            f"Test classes must be ordered as {CLASS_NAMES}, found {ds.class_names}: {test_dir}"
        )
    camera_ids = {Path(path).name.split("_", maxsplit=1)[0] for path in ds.file_paths}
    if camera_ids != {"C009"}:
        raise ValueError(
            f"The held-out benchmark must contain only C009 images, found {sorted(camera_ids)}: {test_dir}"
        )

    y_true = np.concatenate([labels.numpy() for _, labels in ds])
    if len(y_true) != len(ds.file_paths):
        raise ValueError(
            f"Dataset labels and files do not match ({len(y_true)} labels, "
            f"{len(ds.file_paths)} files): {test_dir}"
        )
    return ds.prefetch(tf.data.AUTOTUNE), y_true


def top_class_calibration(
    confidence: np.ndarray,
    correct: np.ndarray,
    bin_count: int = 10,
) -> Tuple[float, list]:
    """Calculate top-label ECE and populated confidence bins."""
    if bin_count < 1:
        raise ValueError("bin_count must be at least 1.")
    if len(confidence) == 0 or len(confidence) != len(correct):
        raise ValueError("Confidence and correctness arrays must be non-empty and have equal length.")

    expected_calibration_error = 0.0
    calibration_bins = []
    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)

    for bin_index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if bin_index == bin_count - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        count = int(np.sum(in_bin))
        if count == 0:
            continue

        mean_score = float(np.mean(confidence[in_bin]))
        empirical_accuracy = float(np.mean(correct[in_bin]))
        expected_calibration_error += (count / len(confidence)) * abs(
            empirical_accuracy - mean_score
        )
        calibration_bins.append({
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "count": count,
            "mean_top_class_score": mean_score,
            "empirical_accuracy": empirical_accuracy,
        })

    return float(expected_calibration_error), calibration_bins


def evaluate_model_on_dataset(
    model: tf.keras.Model,
    ds: tf.data.Dataset,
    y_true: np.ndarray,
) -> Dict[str, Any]:
    """Run inference and compute classification, ordinal, and score-reliability metrics."""
    raw_preds = model.predict(ds, verbose=0)
    if raw_preds.shape != (len(y_true), len(CLASS_NAMES)):
        raise ValueError(
            f"Expected probability output of shape ({len(y_true)}, {len(CLASS_NAMES)}), "
            f"received {raw_preds.shape}."
        )
    if not np.all(np.isfinite(raw_preds)):
        raise ValueError("Model produced non-finite class probabilities.")
    if np.any(raw_preds < 0.0) or np.any(raw_preds > 1.0):
        raise ValueError("Model outputs must be probabilities in the range [0, 1].")
    if not np.allclose(np.sum(raw_preds, axis=1), 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError("Model class probabilities must sum to 1 for every image.")

    y_pred = np.argmax(raw_preds, axis=1)
    if np.any(y_true < 0) or np.any(y_true >= len(CLASS_NAMES)):
        raise ValueError("Ground-truth labels must be okta classes 0 through 8.")

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Ordinal okta error metrics: Mean Absolute Error and Adjacent Accuracy (within +-1 okta)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    adjacent_acc = float(np.mean(np.abs(y_true - y_pred) <= 1))

    # Per-class metrics
    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    per_class = {}
    for i in range(len(CLASS_NAMES)):
        per_class[str(i)] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(s_per[i]),
        }

    top_class_scores = np.max(raw_preds, axis=1)
    correct = y_pred == y_true
    top_class_ece, score_bins = top_class_calibration(top_class_scores, correct)
    one_hot_labels = np.eye(len(CLASS_NAMES), dtype=np.float32)[y_true]
    brier_score = float(np.mean(np.sum(np.square(raw_preds - one_hot_labels), axis=1)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))

    return {
        "sample_count": int(len(y_true)),
        "class_support": {str(i): int(s_per[i]) for i in range(len(CLASS_NAMES))},
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "okta_mae": mae,
        "adjacent_accuracy": adjacent_acc,
        "mean_top_class_score": float(np.mean(top_class_scores)),
        "top_class_ece_10": top_class_ece,
        "multiclass_brier_score": brier_score,
        "top_class_score_bins": score_bins,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred.tolist(),
        "y_true": y_true.tolist(),
    }


def plot_side_by_side_confusion_matrix(
    cm_base: np.ndarray,
    cm_v2: np.ndarray,
    acc_base: float,
    acc_v2: float,
    save_path: Path,
):
    """Plot publication-quality side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#080e19")

    # Dark observatory styling
    for ax, cm, title, acc in zip(
        axes,
        [cm_base, cm_v2],
        ["Baseline (MobileNetV1 Head)", "Zenith V2 (Domain-Augmented)"],
        [acc_base, acc_v2],
    ):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(cmap="magma", values_format="d", ax=ax, colorbar=False)
        ax.set_title(f"{title}\nAccuracy: {acc * 100:.1f}%", color="#f3d39b", fontsize=14, pad=12, fontweight="bold")
        ax.set_xlabel("Predicted Okta (0–8)", color="#eadcc3", fontsize=12)
        ax.set_ylabel("True Okta (0–8)", color="#eadcc3", fontsize=12)
        ax.tick_params(colors="#eadcc3")
        for spine in ax.spines.values():
            spine.set_color("#a9824f")

    plt.suptitle("Zenith Sky Cloud Cover Evaluation (Unseen Camera C009)", color="#e5bd78", fontsize=16, y=0.98, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[OK] Comparison confusion matrix saved: {save_path}")


def run_full_evaluation():
    """Execute complete benchmarking between Baseline and V2 models."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=========================================================")
    print("      ZENITH BENCHMARK EVALUATION: BASELINE vs V2        ")
    print("=========================================================")

    # 1. Load test datasets
    print("Loading held-out test datasets (unseen Camera C009)...")
    ds_night, y_true_night = load_test_dataset("night")
    ds_day, y_true_day = load_test_dataset("day")

    results = {}

    # 2. Evaluate Baseline
    if BASELINE_MODEL_PATH.exists():
        print(f"\nEvaluating Baseline model: {BASELINE_MODEL_PATH} ...")
        model_base = tf.keras.models.load_model(str(BASELINE_MODEL_PATH), compile=False)
        results["baseline_night"] = evaluate_model_on_dataset(model_base, ds_night, y_true_night)
        results["baseline_day"] = evaluate_model_on_dataset(model_base, ds_day, y_true_day)
        print(f"  Baseline Night Accuracy: {results['baseline_night']['accuracy'] * 100:.2f}% (MAE: {results['baseline_night']['okta_mae']:.2f} oktas)")
        print(f"  Baseline Day Accuracy:   {results['baseline_day']['accuracy'] * 100:.2f}% (MAE: {results['baseline_day']['okta_mae']:.2f} oktas)")
    else:
        print(f"[WARNING] Baseline model not found at {BASELINE_MODEL_PATH}")

    # 3. Evaluate V2
    if V2_MODEL_PATH.exists():
        print(f"\nEvaluating Zenith V2 model: {V2_MODEL_PATH} ...")
        model_v2 = tf.keras.models.load_model(str(V2_MODEL_PATH), compile=False)
        results["v2_night"] = evaluate_model_on_dataset(model_v2, ds_night, y_true_night)
        results["v2_day"] = evaluate_model_on_dataset(model_v2, ds_day, y_true_day)
        print(f"  Zenith V2 Night Accuracy: {results['v2_night']['accuracy'] * 100:.2f}% (MAE: {results['v2_night']['okta_mae']:.2f} oktas)")
        print(f"  Zenith V2 Day Accuracy:   {results['v2_day']['accuracy'] * 100:.2f}% (MAE: {results['v2_day']['okta_mae']:.2f} oktas)")
    else:
        print(f"[WARNING] V2 model not found at {V2_MODEL_PATH}")

    # 4. Generate comparison plots
    if "baseline_night" in results and "v2_night" in results:
        cm_base = np.array(results["baseline_night"]["confusion_matrix"])
        cm_v2 = np.array(results["v2_night"]["confusion_matrix"])
        plot_side_by_side_confusion_matrix(
            cm_base,
            cm_v2,
            results["baseline_night"]["accuracy"],
            results["v2_night"]["accuracy"],
            REPORTS_DIR / "confusion_matrix_comparison.png",
        )

    # 5. Save single V2 confusion matrix
    if "v2_night" in results:
        cm_v2 = np.array(results["v2_night"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 7), facecolor="#080e19")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_v2, display_labels=CLASS_NAMES)
        disp.plot(cmap="magma", values_format="d", ax=ax, colorbar=False)
        ax.set_title(f"Zenith V2 — Nighttime Confusion Matrix\nAccuracy: {results['v2_night']['accuracy']*100:.1f}%", color="#f3d39b", fontsize=13, pad=10, fontweight="bold")
        ax.set_xlabel("Predicted Okta (0–8)", color="#eadcc3", fontsize=11)
        ax.set_ylabel("True Okta (0–8)", color="#eadcc3", fontsize=11)
        ax.tick_params(colors="#eadcc3")
        for spine in ax.spines.values():
            spine.set_color("#a9824f")
        plt.tight_layout()
        plt.savefig(str(REPORTS_DIR / "confusion_matrix_v2.png"), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()

    # 6. Save JSON evaluation metrics
    clean_results = {}
    for key, val in results.items():
        clean_results[key] = {
            "sample_count": val["sample_count"],
            "class_support": val["class_support"],
            "accuracy": val["accuracy"],
            "macro_precision": val["macro_precision"],
            "macro_recall": val["macro_recall"],
            "macro_f1": val["macro_f1"],
            "weighted_f1": val["weighted_f1"],
            "okta_mae": val["okta_mae"],
            "adjacent_accuracy": val["adjacent_accuracy"],
            "mean_top_class_score": val["mean_top_class_score"],
            "top_class_ece_10": val["top_class_ece_10"],
            "multiclass_brier_score": val["multiclass_brier_score"],
            "top_class_score_bins": val["top_class_score_bins"],
            "per_class": val["per_class"],
        }
    with open(REPORTS_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)

    # 7. Generate Model Evaluation Markdown Report
    generate_markdown_report(results)
    print("\n[SUCCESS] Full evaluation complete. Results written to reports/")


def generate_markdown_report(results: Dict[str, Any]):
    """Generate comprehensive model evaluation and baseline comparison markdown report."""
    rep_path = REPORTS_DIR / "model_evaluation.md"

    base_night = results.get("baseline_night", {})
    v2_night = results.get("v2_night", {})
    base_day = results.get("baseline_day", {})
    v2_day = results.get("v2_day", {})

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Zenith Sky Condition Monitoring — Model Evaluation Report (V2)\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("This report presents the rigorous evaluation of the **Zenith V2 Sky Condition Monitoring Model** ")
        f.write("against the original baseline architecture on the held-out test dataset from unseen all-sky camera `C009`.\n\n")

        f.write("### Benchmark Comparison (Held-Out Test Set — Unseen Camera C009)\n\n")
        f.write("| Evaluation Metric | Baseline Model (`zenith_mobilenetv2.keras`) | Zenith V2 (`zenith_v2.keras`) | Delta / Improvement |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")

        b_acc = base_night.get("accuracy", 0.0) * 100
        v_acc = v2_night.get("accuracy", 0.0) * 100
        f.write(f"| **Nighttime Accuracy** | {b_acc:.2f}% | **{v_acc:.2f}%** | **{v_acc - b_acc:+.2f}%** |\n")

        b_mae = base_night.get("okta_mae", 0.0)
        v_mae = v2_night.get("okta_mae", 0.0)
        f.write(f"| **Nighttime Okta MAE** | {b_mae:.2f} oktas | **{v_mae:.2f} oktas** | **{v_mae - b_mae:+.2f} oktas** |\n")

        b_adj = base_night.get("adjacent_accuracy", 0.0) * 100
        v_adj = v2_night.get("adjacent_accuracy", 0.0) * 100
        f.write(f"| **Within ±1 Okta Accuracy** | {b_adj:.2f}% | **{v_adj:.2f}%** | **{v_adj - b_adj:+.2f}%** |\n")

        b_f1 = base_night.get("macro_f1", 0.0) * 100
        v_f1 = v2_night.get("macro_f1", 0.0) * 100
        f.write(f"| **Macro F1-Score (Night)** | {b_f1:.2f}% | **{v_f1:.2f}%** | **{v_f1 - b_f1:+.2f}%** |\n")

        b_day_acc = base_day.get("accuracy", 0.0) * 100
        v_day_acc = v2_day.get("accuracy", 0.0) * 100
        f.write(f"| **Daytime Accuracy (C009)** | {b_day_acc:.2f}% | **{v_day_acc:.2f}%** | **{v_day_acc - b_day_acc:+.2f}%** |\n\n")
        f.write(
            f"Nighttime test sample counts: baseline **{base_night.get('sample_count', 0)}**, "
            f"V2 **{v2_night.get('sample_count', 0)}**. Per-class support (V2): "
            f"`{v2_night.get('class_support', {})}`.\n\n"
        )

        f.write("## 2. Key Architectural & Training Enhancements in V2\n\n")
        f.write("1. **Domain-Generalizing Data Augmentation**: Injected rotational symmetry, brightness shifts (±40), contrast jitter, and 30% desaturation/grayscale conversion to prevent the CNN from relying on daytime blue-sky chromaticity.\n")
        f.write("2. **Nighttime Adaptation Set Integration**: Holds out 20% of C005 nighttime images for validation and oversamples the remaining images for training.\n")
        f.write("3. **Two-Stage Controlled Transfer Learning**: Phase 1 trained a regularized head (GAP + BatchNorm + Dense(256) + L2 + Dropout(0.4)), followed by Phase 2 controlled fine-tuning of the top 35 layers with gentle learning rate (`7e-5`) and early stopping.\n")
        f.write("4. **Continuous Ordinal Prediction**: Expected Okta calculation ($\\sum c \\cdot P(c)$) complements discrete classification for fine-grained astronomical windowing.\n\n")

        f.write("## 3. Detailed Class-Wise Performance Breakdown (Nighttime Test Set)\n\n")
        f.write("| Okta Class | Cloud Cover Category | Support | V2 Precision | V2 Recall | V2 F1-Score | Baseline F1-Score |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")

        v2_per = v2_night.get("per_class", {})
        base_per = base_night.get("per_class", {})
        descriptions = [
            "0 (Completely Clear)", "1 (Nearly Clear)", "2 (Scattered)", "3 (Scattered)",
            "4 (Half Clouded)", "5 (Broken)", "6 (Broken)", "7 (Nearly Overcast)", "8 (Overcast)"
        ]

        for i in range(9):
            vp = v2_per.get(str(i), {})
            bp = base_per.get(str(i), {})
            f.write(
                f"| **{i}** | {descriptions[i]} | "
                f"{vp.get('support', 0)} | "
                f"{vp.get('precision', 0.0):.3f} | "
                f"{vp.get('recall', 0.0):.3f} | "
                f"{vp.get('f1', 0.0):.3f} | "
                f"{bp.get('f1', 0.0):.3f} |\n"
            )

        f.write("\n## 4. Astronomical Suitability & Measurable Image Indicators\n\n")
        f.write("- **Image Indicators**: Zenith calculates brightness, contrast, saturation, and Laplacian sharpness.\n")
        f.write("- **Physical Disclaimers**: Zenith makes explicit that RGB indicators evaluate optical camera conditions, not atmospheric seeing (arcsecond turbulence) or photometer sky background (mag/arcsec²).\n")
        f.write("\n## 5. Top-Class Score Reliability\n\n")
        f.write("| Nighttime Metric | Baseline | Zenith V2 |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(
            f"| Exact accuracy | {base_night.get('accuracy', 0.0):.3f} | "
            f"{v2_night.get('accuracy', 0.0):.3f} |\n"
        )
        f.write(
            f"| Mean top-class softmax score | {base_night.get('mean_top_class_score', 0.0):.3f} | "
            f"{v2_night.get('mean_top_class_score', 0.0):.3f} |\n"
        )
        f.write(
            f"| Top-class ECE (10 bins; lower is better) | {base_night.get('top_class_ece_10', 0.0):.3f} | "
            f"{v2_night.get('top_class_ece_10', 0.0):.3f} |\n"
        )
        f.write(
            f"| Multiclass Brier score (lower is better) | {base_night.get('multiclass_brier_score', 0.0):.3f} | "
            f"{v2_night.get('multiclass_brier_score', 0.0):.3f} |\n\n"
        )
        f.write(
            "The top-class softmax score is not accuracy or a calibrated probability. ECE compares "
            "mean top-class scores with observed correctness in equal-width confidence bins; "
            "the Brier score evaluates the full probability vector.\n\n"
        )
        v2_score_gap = v2_night.get("mean_top_class_score", 0.0) - v2_night.get("accuracy", 0.0)
        if v2_score_gap > 0:
            f.write(
                f"On this nighttime test set, the V2 mean top-class score exceeds exact accuracy "
                f"by **{v2_score_gap * 100:.1f} percentage points**, indicating substantial "
                "overconfidence on average.\n\n"
            )
        f.write("| V2 top-class score range | Images | Mean score | Observed exact accuracy |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for score_bin in v2_night.get("top_class_score_bins", []):
            f.write(
                f"| {score_bin['lower_bound']:.1f}–{score_bin['upper_bound']:.1f} | "
                f"{score_bin['count']} | {score_bin['mean_top_class_score']:.3f} | "
                f"{score_bin['empirical_accuracy']:.3f} |\n"
            )
        f.write("\n")
        f.write("## 6. Generalization Limits\n\n")
        f.write(
            "The V2 nighttime result on unseen camera C009 is "
            f"**{v_acc:.2f}% exact accuracy**, **{v_adj:.2f}% within ±1 okta**, "
            f"and **{v_mae:.2f} okta MAE**. Exact-class accuracy alone understates "
            "the ordinal nature of some errors, but the result is not strong evidence "
            "of reliable performance on arbitrary cameras.\n\n"
        )
        low_recall_classes = [
            str(index)
            for index in range(9)
            if v2_night.get("per_class", {}).get(str(index), {}).get("recall", 1.0) < 0.1
        ]
        if low_recall_classes:
            f.write(
                "Some classes remain especially unreliable: nighttime recall is below "
                f"10% for okta class(es) {', '.join(low_recall_classes)}.\n\n"
            )
        f.write(
            "The training sources include 5,400 daytime images from C013 and only "
            "180 nighttime images from C005. The revised pipeline holds out 36 C005 "
            "images for checkpoint validation and oversamples the remaining 144; "
            "because both sets come from C005, this is not unseen-camera validation. "
            "The former C013 validation score also did not estimate cross-camera "
            "performance. C009 remains held out for final evaluation. This covers "
            "only a small number of camera domains and does not validate ordinary "
            "photos or other sensor types. Predictions should be treated as estimates "
            "until evaluated on labeled data from the intended camera.\n"
        )


if __name__ == "__main__":
    run_full_evaluation()
