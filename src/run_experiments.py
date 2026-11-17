from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
LABELS = np.arange(10)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
    values = np.loadtxt(path, delimiter=",", dtype=np.int16)
    if values.ndim != 2 or values.shape[1] != 65:
        raise ValueError(f"{path.name} phải có 65 cột, nhận được {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} chứa giá trị không hữu hạn")

    features = values[:, :-1]
    labels = values[:, -1]
    if features.min() < 0 or features.max() > 16:
        raise ValueError(f"{path.name}: pixel phải nằm trong [0, 16]")
    if not np.isin(labels, LABELS).all():
        raise ValueError(f"{path.name}: nhãn phải thuộc các chữ số 0–9")
    return features, labels


def hog_features(features: np.ndarray) -> np.ndarray:
    images = features.reshape(-1, 8, 8).astype(np.float32) / 16.0
    orientations = 9
    bin_width = 180.0 / orientations
    descriptors = []
    for image in images:
        gradient_y, gradient_x = np.gradient(image)
        magnitude = np.hypot(gradient_x, gradient_y)
        angle = np.mod(np.rad2deg(np.arctan2(gradient_y, gradient_x)), 180.0)
        bin_position = angle / bin_width
        lower_bin = np.floor(bin_position).astype(np.intp) % orientations
        upper_bin = (lower_bin + 1) % orientations
        upper_weight = bin_position - np.floor(bin_position)
        lower_weight = 1.0 - upper_weight

        cells = np.zeros((4, 4, orientations), dtype=np.float32)
        for cell_y in range(4):
            for cell_x in range(4):
                row_slice = slice(cell_y * 2, (cell_y + 1) * 2)
                col_slice = slice(cell_x * 2, (cell_x + 1) * 2)
                cell_magnitude = magnitude[row_slice, col_slice].ravel()
                cell_lower = lower_bin[row_slice, col_slice].ravel()
                cell_upper = upper_bin[row_slice, col_slice].ravel()
                np.add.at(
                    cells[cell_y, cell_x],
                    cell_lower,
                    cell_magnitude * lower_weight[row_slice, col_slice].ravel(),
                )
                np.add.at(
                    cells[cell_y, cell_x],
                    cell_upper,
                    cell_magnitude * upper_weight[row_slice, col_slice].ravel(),
                )

        blocks = []
        for block_y in range(3):
            for block_x in range(3):
                block = cells[block_y : block_y + 2, block_x : block_x + 2].ravel()
                block = block / np.sqrt(np.dot(block, block) + 1e-5**2)
                block = np.minimum(block, 0.2)
                block = block / np.sqrt(np.dot(block, block) + 1e-5**2)
                blocks.extend(block)
        descriptors.append(blocks)
    return np.asarray(descriptors, dtype=np.float32)


def metric_values(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision_macro": float(
            precision_score(expected, predicted, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(expected, predicted, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(expected, predicted, average="macro", zero_division=0)
        ),
    }


def tune_knn(
    x_fit: np.ndarray, y_fit: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for metric in ("euclidean", "manhattan"):
        for k in (1, 3, 5, 7, 9, 11):
            model = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            model.fit(x_fit, y_fit)
            score = metric_values(y_val, model.predict(x_val))
            trial = {
                "model": "KNN",
                "metric": metric,
                "n_neighbors": k,
                "max_depth": "",
                "min_samples_leaf": "",
                **score,
            }
            trials.append(trial)
            if best is None or (score["f1_macro"], score["accuracy"]) > (
                best["validation_f1_macro"],
                best["validation_accuracy"],
            ):
                best = {
                    "model": "KNN",
                    "metric": metric,
                    "n_neighbors": k,
                    "validation_f1_macro": score["f1_macro"],
                    "validation_accuracy": score["accuracy"],
                }
    if best is None:
        raise RuntimeError("Không tìm được cấu hình K-NN")
    return best, trials


def tune_tree(
    x_fit: np.ndarray, y_fit: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for depth in (None, 5, 10, 15, 20, 30):
        for leaf_size in (1, 2, 4):
            model = DecisionTreeClassifier(
                criterion="gini",
                max_depth=depth,
                min_samples_leaf=leaf_size,
                random_state=seed,
            )
            model.fit(x_fit, y_fit)
            score = metric_values(y_val, model.predict(x_val))
            trial = {
                "model": "Decision Tree",
                "metric": "",
                "n_neighbors": "",
                "max_depth": "None" if depth is None else depth,
                "min_samples_leaf": leaf_size,
                **score,
            }
            trials.append(trial)
            if best is None or (score["f1_macro"], score["accuracy"]) > (
                best["validation_f1_macro"],
                best["validation_accuracy"],
            ):
                best = {
                    "model": "Decision Tree",
                    "max_depth": depth,
                    "min_samples_leaf": leaf_size,
                    "validation_f1_macro": score["f1_macro"],
                    "validation_accuracy": score["accuracy"],
                }
    if best is None:
        raise RuntimeError("Không tìm được cấu hình Decision Tree")
    return best, trials


def save_eda(
    x_train: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, output: Path
) -> None:
    counts_train = np.bincount(y_train, minlength=10)
    counts_test = np.bincount(y_test, minlength=10)
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(10)
    ax.bar(positions - 0.2, counts_train, width=0.4, label="Train")
    ax.bar(positions + 0.2, counts_test, width=0.4, label="Test")
    ax.set(title="Phân phối nhãn theo chữ số", xlabel="Nhãn", ylabel="Số mẫu")
    ax.set_xticks(positions)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for digit, ax in enumerate(axes.flat):
        index = int(np.flatnonzero(y_train == digit)[0])
        ax.imshow(x_train[index].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        ax.set_title(f"Chữ số {digit}")
        ax.axis("off")
    fig.suptitle("Một mẫu đại diện cho mỗi lớp")
    fig.tight_layout()
    fig.savefig(output / "sample_digits.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Tóm tắt EDA",
        "",
        f"- Tập train: **{len(y_train):,}** mẫu, 64 đặc trưng pixel và 1 nhãn.",
        f"- Tập test chính thức: **{len(y_test):,}** mẫu.",
        "- Ảnh được biểu diễn thành ma trận 8×8; pixel nguyên trong khoảng 0–16.",
        "- Nhãn là chữ số 0–9. Dữ liệu CSV không có header.",
        "- Giá trị thiếu: 0; mọi pixel và nhãn đều được kiểm tra miền hợp lệ khi nạp.",
        f"- Tổng số bản ghi trùng hoàn toàn trong train: {len(y_train) - len(np.unique(np.column_stack((x_train, y_train)), axis=0))}.",
        "",
        "## Số mẫu theo lớp",
        "",
        "| Chữ số | Train | Test |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {digit} | {counts_train[digit]} | {counts_test[digit]} |"
        for digit in range(10)
    )
    (output / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Không có kết quả để ghi vào {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results_tex(
    output_dir: Path, rows: list[dict[str, Any]], selected_name: str, configs: dict[str, dict[str, Any]]
) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Phương án & Tham số & Accuracy & Precision & Recall & Macro F1 \\",
        r"\midrule",
    ]
    for row in rows:
        config = configs[row["experiment"]]
        if config["model"] == "KNN":
            parameters = f"k={config['n_neighbors']}, {config['metric']}"
        else:
            depth = config["max_depth"]
            parameters = (
                f"depth={'không giới hạn' if depth is None else depth}, "
                f"leaf={config['min_samples_leaf']}"
            )
        selected_marker = r"\textbf{*} " if row["experiment"] == selected_name else ""
        lines.append(
            f"{selected_marker}{row['experiment']} & {parameters} & "
            f"{row['accuracy']:.4f} & {row['precision_macro']:.4f} & "
            f"{row['recall_macro']:.4f} & {row['f1_macro']:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Kết quả trên test set chính thức. Dấu * đánh dấu mô hình được chọn bằng validation.}",
            r"\label{tab:results}",
            r"\end{table}",
        ]
    )
    (output_dir / "results_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(data_dir: Path, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_dataset(data_dir / "optdigits.tra")
    x_test, y_test = load_dataset(data_dir / "optdigits.tes")
    save_eda(x_train, y_train, y_test, output_dir)

    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train,
    )
    representations = {
        "Pixel + KNN": (x_train.astype(np.float32) / 16.0, x_fit / 16.0, x_val / 16.0, x_test / 16.0),
        "HOG + KNN": (
            hog_features(x_train),
            hog_features(x_fit),
            hog_features(x_val),
            hog_features(x_test),
        ),
    }
    tuned: dict[str, dict[str, Any]] = {}
    tuning_rows: list[dict[str, Any]] = []
    for representation, (_, fit_x, val_x, _) in representations.items():
        best, trials = tune_knn(fit_x, y_fit, val_x, y_val)
        tuned[representation] = best
        tuning_rows.extend({"representation": representation, **row} for row in trials)

    pixel_best, pixel_trials = tune_tree(
        representations["Pixel + KNN"][1],
        y_fit,
        representations["Pixel + KNN"][2],
        y_val,
        seed,
    )
    hog_best, hog_trials = tune_tree(
        representations["HOG + KNN"][1],
        y_fit,
        representations["HOG + KNN"][2],
        y_val,
        seed,
    )
    tuned["Pixel + Decision Tree"] = pixel_best
    tuned["HOG + Decision Tree"] = hog_best
    tuning_rows.extend(
        {"representation": "Pixel + Decision Tree", **row} for row in pixel_trials
    )
    tuning_rows.extend(
        {"representation": "HOG + Decision Tree", **row} for row in hog_trials
    )
    save_csv(output_dir / "tuning_results.csv", tuning_rows)

    result_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    x_test_pixels = x_test.astype(np.float32) / 16.0
    x_test_hog = representations["HOG + KNN"][3]
    full_features = {
        "Pixel + KNN": (x_train.astype(np.float32) / 16.0, x_test_pixels),
        "HOG + KNN": (representations["HOG + KNN"][0], x_test_hog),
        "Pixel + Decision Tree": (x_train.astype(np.float32) / 16.0, x_test_pixels),
        "HOG + Decision Tree": (representations["HOG + KNN"][0], x_test_hog),
    }
    for name, config in tuned.items():
        fit_x, test_x = full_features[name]
        if config["model"] == "KNN":
            model = KNeighborsClassifier(
                n_neighbors=config["n_neighbors"],
                metric=config["metric"],
                n_jobs=-1,
            )
        else:
            model = DecisionTreeClassifier(
                criterion="gini",
                max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"],
                random_state=seed,
            )
        model.fit(fit_x, y_train)
        predicted = model.predict(test_x)
        test_predictions[name] = predicted
        fitted_models[name] = model
        result_rows.append(
            {
                "experiment": name,
                "selected_parameters": json.dumps(config, ensure_ascii=False),
                **metric_values(y_test, predicted),
            }
        )
    save_csv(output_dir / "metrics.csv", result_rows)

    chosen = max(
        tuned,
        key=lambda name: (
            tuned[name]["validation_f1_macro"],
            tuned[name]["validation_accuracy"],
        ),
    )
    save_results_tex(output_dir, result_rows, chosen, tuned)
    best_model = fitted_models[chosen]
    best_prediction = test_predictions[chosen]
    joblib.dump(best_model, output_dir / "best_model.joblib")
    (output_dir / "best_model.json").write_text(
        json.dumps(
            {
                "experiment": chosen,
                "parameters": tuned[chosen],
                "validation_split": 0.2,
                "random_state": seed,
                "official_train_samples": len(y_train),
                "official_test_samples": len(y_test),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_test, best_prediction, labels=LABELS)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"Confusion matrix — {chosen}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    names = [row["experiment"] for row in result_rows]
    positions = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, metric in enumerate(
        ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    ):
        ax.bar(
            positions + (index - 1.5) * width,
            [row[metric] for row in result_rows],
            width,
            label=metric.replace("_", " "),
        )
    ax.set_xticks(positions, names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Điểm số")
    ax.set_title("So sánh mô hình trên test set chính thức")
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_comparison.png", dpi=160)
    plt.close(fig)

    errors = [
        {
            "index_in_official_test": int(index),
            "true_label": int(y_test[index]),
            "predicted_label": int(best_prediction[index]),
        }
        for index in np.flatnonzero(best_prediction != y_test)
    ]
    save_csv(
        output_dir / "errors.csv",
        errors
        or [{"index_in_official_test": "", "true_label": "", "predicted_label": ""}],
    )
    error_counts = Counter(
        (error["true_label"], error["predicted_label"]) for error in errors
    )
    top_confusions = error_counts.most_common(5)
    report_lines = [
        f"Mô hình được chọn bằng validation macro F1: {chosen}",
        "",
        classification_report(
            y_test, best_prediction, labels=LABELS, zero_division=0, digits=4
        ),
        "Các cặp nhầm lẫn phổ biến nhất (nhãn thật -> dự đoán):",
        *(
            f"{true_label} -> {predicted_label}: {count}"
            for (true_label, predicted_label), count in top_confusions
        ),
        f"Tổng số mẫu test dự đoán sai: {len(errors)} / {len(y_test)}",
    ]
    (output_dir / "error_analysis.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    for ax, error in zip(axes.flat, errors[:12]):
        index = error["index_in_official_test"]
        ax.imshow(x_test[index].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        ax.set_title(f"Thật {y_test[index]} / đoán {best_prediction[index]}")
        ax.axis("off")
    for ax in axes.flat[len(errors[:12]) :]:
        ax.axis("off")
    fig.suptitle(f"Một số lỗi của mô hình được chọn ({chosen})")
    fig.tight_layout()
    fig.savefig(output_dir / "misclassified_samples.png", dpi=160)
    plt.close(fig)

    print(f"Train: {x_train.shape}; test: {x_test.shape}")
    print(f"Chọn bằng validation: {chosen}; tham số: {tuned[chosen]}")
    for row in result_rows:
        print(
            f"{row['experiment']}: accuracy={row['accuracy']:.4f}, "
            f"macro-F1={row['f1_macro']:.4f}"
        )
    print(f"Đã lưu kết quả tại: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm nhận dạng chữ số.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
