"""Phân loại táo, chuối, cam bằng đặc trưng thống kê và màu sắc.

Cấu trúc dữ liệu mong đợi:
    dataset_bai1/
        apple/ *.jpg|png|...
        banana/ ...
        orange/ ...

Cài thư viện: pip install opencv-python numpy pandas scikit-learn matplotlib
Chạy: python fruit_classification.py --data-dir dataset_bai1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (128, 128)  # OpenCV dùng thứ tự (width, height).
HIST_BINS = 8


def entropy_grayscale(rgb: np.ndarray) -> float:
    """Entropy Shannon (bit) của histogram 256 mức xám."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    counts = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def extract_features(rgb: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Trả về các nhóm đặc trưng và ảnh HSV (uint8) tương ứng."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    result: Dict[str, float] = {}

    for space_name, image in (("RGB", rgb), ("HSV", hsv)):
        for channel, channel_name in enumerate(("C1", "C2", "C3")):
            plane = image[:, :, channel].astype(np.float32)
            result[f"{space_name}_{channel_name}_mean"] = float(plane.mean())
            result[f"{space_name}_{channel_name}_std"] = float(plane.std())

    # Histogram HSV theo từng kênh; mỗi kênh được chuẩn hóa tổng bằng 1.
    for channel, channel_name in enumerate(("H", "S", "V")):
        # OpenCV lưu Hue trong khoảng [0, 180), còn Saturation/Value là [0, 256).
        upper_bound = 180 if channel == 0 else 256
        hist = cv2.calcHist([hsv], [channel], None, [HIST_BINS], [0, upper_bound]).ravel()
        hist = hist / max(float(hist.sum()), 1.0)
        for bin_index, value in enumerate(hist):
            result[f"HSV_hist_{channel_name}_{bin_index}"] = float(value)

    result["gray_entropy"] = entropy_grayscale(rgb)
    return result, hsv


def read_dataset(data_dir: Path) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

    rows: List[dict] = []
    rgb_images: List[np.ndarray] = []
    hsv_images: List[np.ndarray] = []
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_dirs:
        raise ValueError(f"Không tìm thấy thư mục lớp con trong {data_dir}")

    for class_dir in class_dirs:
        image_paths = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Cảnh báo: bỏ qua ảnh không đọc được: {image_path}")
                continue
            bgr = cv2.resize(bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            features, hsv = extract_features(rgb)
            features.update({"label": class_dir.name, "file": str(image_path)})
            rows.append(features)
            rgb_images.append(rgb)
            hsv_images.append(hsv)

    if not rows:
        raise ValueError(f"Không đọc được ảnh nào trong {data_dir}")
    return pd.DataFrame(rows), rgb_images, hsv_images


def show_examples(frame: pd.DataFrame, rgb_images: List[np.ndarray], hsv_images: List[np.ndarray], output: Path,
                  show: bool) -> None:
    """Lưu/hiển thị minh họa RGB sau resize và chuyển đổi RGB sang HSV."""
    labels = frame["label"].tolist()
    unique_labels = list(dict.fromkeys(labels))
    selected = [labels.index(label) for label in unique_labels]
    fig, axes = plt.subplots(len(selected), 2, figsize=(7, max(3, 2.7 * len(selected))), squeeze=False)
    for row, image_index in enumerate(selected):
        axes[row, 0].imshow(rgb_images[image_index])
        axes[row, 0].set_title(f"{labels[image_index]} — RGB 128×128")
        # HSV không phải màu hiển thị trực quan; chuyển lại RGB để minh họa đúng ảnh.
        hsv_as_rgb = cv2.cvtColor(hsv_images[image_index], cv2.COLOR_HSV2RGB)
        axes[row, 1].imshow(hsv_as_rgb)
        axes[row, 1].set_title("Sau chuyển RGB → HSV (hiển thị lại)")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "minh_hoa_tien_xu_ly.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrices(results: dict, labels: List[str], output: Path, show: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (name, result) in zip(axes.ravel(), results.items()):
        matrix = result["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Dự đoán")
        axis.set_ylabel("Thực tế")
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Số ảnh")
    fig.savefig(output / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_experiments(frame: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, dict, List[str], np.ndarray, dict]:
    y = frame["label"].astype(str)
    labels = sorted(y.unique().tolist())
    rgb_cols = [c for c in frame if c.startswith("RGB_")]
    hsv_cols = [c for c in frame if c.startswith("HSV_C")]
    hist_cols = [c for c in frame if c.startswith("HSV_hist_")]
    entropy_cols = ["gray_entropy"]
    experiments = {
        "TN1 - Mean+Std RGB / KNN": (rgb_cols, KNeighborsClassifier(n_neighbors=5)),
        "TN2 - Mean+Std HSV / KNN": (hsv_cols, KNeighborsClassifier(n_neighbors=5)),
        "TN3 - Histogram HSV / SVM": (hist_cols, SVC(kernel="rbf", C=1.0, gamma="scale")),
        "TN4 - Toàn bộ / SVM": (rgb_cols + hsv_cols + hist_cols + entropy_cols,
                                  SVC(kernel="rbf", C=1.0, gamma="scale")),
    }
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    records = []
    results = {}
    predictions_by_experiment = {}
    for name, (columns, estimator) in experiments.items():
        X_train = frame.iloc[train_idx][columns].to_numpy(dtype=np.float64)
        X_test = frame.iloc[test_idx][columns].to_numpy(dtype=np.float64)
        # Pipeline đảm bảo scaler chỉ fit trên tập train.
        model = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        model.fit(X_train, y.iloc[train_idx])
        prediction = model.predict(X_test)
        matrix = confusion_matrix(y.iloc[test_idx], prediction, labels=labels)
        metrics = {
            "Accuracy": accuracy_score(y.iloc[test_idx], prediction),
            "Precision_macro": precision_score(y.iloc[test_idx], prediction, average="macro", zero_division=0),
            "Recall_macro": recall_score(y.iloc[test_idx], prediction, average="macro", zero_division=0),
            "F1_macro": f1_score(y.iloc[test_idx], prediction, average="macro", zero_division=0),
        }
        records.append({"Thí nghiệm": name, **metrics})
        results[name] = {
            "confusion_matrix": matrix,
            "report": classification_report(y.iloc[test_idx], prediction, labels=labels,
                                             target_names=labels, zero_division=0),
        }
        predictions_by_experiment[name] = prediction
    return pd.DataFrame(records), results, labels, test_idx, predictions_by_experiment


def save_prediction_images(frame: pd.DataFrame, rgb_images: List[np.ndarray], test_idx: np.ndarray,
                           predictions: dict, output: Path, show: bool = False) -> pd.DataFrame:
    """Lưu nhãn dự đoán cho từng ảnh test và contact sheet cho TN4."""
    tn4_name = "TN4 - Toàn bộ / SVM"
    tn4_predictions = predictions[tn4_name]
    rows = []
    for exp_name, exp_predictions in predictions.items():
        for index, predicted in zip(test_idx, exp_predictions):
            rows.append({"file": frame.iloc[index]["file"], "label_thuc_te": frame.iloc[index]["label"],
                         "thi_nghiem": exp_name, "label_du_doan": predicted,
                         "dung": bool(frame.iloc[index]["label"] == predicted)})
    prediction_frame = pd.DataFrame(rows)
    prediction_frame.to_csv(output / "du_doan_tap_test.csv", index=False, encoding="utf-8-sig")

    # Chia thành nhiều trang để hình không quá dài khi test set lớn.
    page_size = 30
    for page, start in enumerate(range(0, len(test_idx), page_size), start=1):
        page_indices = test_idx[start:start + page_size]
        page_preds = tn4_predictions[start:start + page_size]
        ncols = 5
        nrows = int(np.ceil(len(page_indices) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.8 * nrows), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, index, predicted in zip(axes.ravel(), page_indices, page_preds):
            axis.imshow(rgb_images[index])
            actual = str(frame.iloc[index]["label"])
            axis.set_title(f"Dự đoán: {predicted}\nThực tế: {actual}", fontsize=9,
                           color="green" if predicted == actual else "red")
            axis.axis("off")
        fig.suptitle(f"Dự đoán TN4 (SVM) trên ảnh test — trang {page}")
        fig.tight_layout()
        fig.savefig(output / f"anh_du_doan_TN4_{page:02d}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    return prediction_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân loại trái cây với đặc trưng RGB/HSV.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset_bai1"), help="Thư mục dữ liệu, mỗi lớp là một thư mục con.")
    parser.add_argument("--output-dir", type=Path, default=Path("ket_qua_bai1"), help="Thư mục lưu CSV và hình minh họa.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tỷ lệ test, mặc định 0.2 (80/20).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-show", action="store_true", help="Không bật cửa sổ hiển thị biểu đồ.")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size phải nằm giữa 0 và 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, rgb_images, hsv_images = read_dataset(args.data_dir)
    frame.to_csv(args.output_dir / "dac_trung.csv", index=False, encoding="utf-8-sig")
    print(f"Đã trích xuất {len(frame)} ảnh; số ảnh theo lớp:\n{frame['label'].value_counts().to_string()}")
    show_examples(frame, rgb_images, hsv_images, args.output_dir, show=not args.no_show)

    try:
        summary, results, labels, test_idx, predictions = run_experiments(frame, args.test_size, args.random_state)
    except ValueError as exc:
        raise SystemExit(
            "Không thể chia dữ liệu phân tầng hoặc huấn luyện. Hãy đảm bảo mỗi lớp có đủ ảnh "
            "để xuất hiện trong cả train và test. Chi tiết: " + str(exc)
        ) from exc

    summary.to_csv(args.output_dir / "so_sanh_thi_nghiem.csv", index=False, encoding="utf-8-sig")
    print("\n=== BẢNG SO SÁNH (Precision/Recall/F1 là macro) ===")
    print(summary.to_string(index=False, formatters={c: (lambda value: f"{value:.4f}")
                                                     for c in summary.columns if c != "Thí nghiệm"}))
    for name, result in results.items():
        print(f"\n=== {name} ===")
        print("Confusion matrix (hàng = thực tế, cột = dự đoán), thứ tự lớp:", labels)
        print(result["confusion_matrix"])
        print(result["report"])
    plot_confusion_matrices(results, labels, args.output_dir, show=not args.no_show)
    save_prediction_images(frame, rgb_images, test_idx, predictions, args.output_dir)
    write_report(summary, results, labels, frame, args.data_dir, args.output_dir)

    best = summary.loc[summary["F1_macro"].idxmax()]
    print("\n=== NHẬN XÉT TỰ ĐỘNG ===")
    print(f"Theo F1 macro, thí nghiệm tốt nhất trên lần chia này là '{best['Thí nghiệm']}' "
          f"(F1={best['F1_macro']:.4f}, Accuracy={best['Accuracy']:.4f}).")
    print("RGB mean/std mô tả màu trung bình và độ phân tán theo kênh; HSV mean/std tách sắc độ, "
          "độ bão hòa và độ sáng nên thường ít trộn lẫn sắc độ với độ sáng. Histogram HSV giữ "
          "phân bố màu chi tiết hơn nhưng bỏ qua vị trí không gian của màu. Entropy xám mô tả độ "
          "phức tạp/đa dạng mức sáng, không trực tiếp biểu diễn màu. TN4 kết hợp các tín hiệu này; "
          "kết quả phụ thuộc dữ liệu, số lượng mẫu, nền ảnh và lần chia train/test, nên không mặc định "
          "nhóm đặc trưng nào luôn tốt nhất.")
    print(f"\nTệp kết quả đã lưu tại: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
