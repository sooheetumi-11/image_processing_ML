"""Phân loại hình tròn, hình vuông và hình tam giác bằng Shape + HOG.

Cài đặt:
    pip install opencv-python numpy pandas scikit-learn scikit-image matplotlib

Chạy toàn bộ quy trình:
    python shape_hog_classification.py

Ảnh sinh ra, bảng đặc trưng, báo cáo và biểu đồ được lưu trong thư mục
`ket_qua_shape_hog/` (có thể đổi bằng --output-dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSES = ("circle", "square", "triangle")
CLASS_NAMES_VI = {"circle": "Hình tròn", "square": "Hình vuông", "triangle": "Hình tam giác"}
IMAGE_SIZE = 128
HOG_SIZE = (64, 64)
SHAPE_FEATURE_NAMES = ["area", "perimeter", "aspect_ratio", "circularity", "edge_density", "solidity"]
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)


def _rotate_points(points: np.ndarray, center: Tuple[float, float], angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return ((points - np.asarray(center)) @ rotation.T + np.asarray(center)).round().astype(np.int32)


def make_shape_image(label: str, rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    """Sinh ảnh xám nền sáng, vật thể tối; có jitter vị trí/kích thước/góc quay."""
    if label not in CLASSES:
        raise ValueError(f"Nhãn không hợp lệ: {label}")
    image = np.full((size, size), int(rng.integers(225, 256)), dtype=np.uint8)
    center = (int(size / 2 + rng.integers(-15, 16)), int(size / 2 + rng.integers(-15, 16)))
    radius_or_half = int(rng.integers(20, 39))
    angle = float(rng.uniform(0, 360))
    color = int(rng.integers(0, 36))

    if label == "circle":
        cv2.circle(image, center, radius_or_half, color, thickness=-1, lineType=cv2.LINE_AA)
    elif label == "square":
        half = radius_or_half
        points = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)
        points = _rotate_points(points, (0, 0), angle) + np.asarray(center, dtype=np.int32)
        cv2.fillPoly(image, [points], color, lineType=cv2.LINE_AA)
    else:
        # Tam giác cân với đáy và đỉnh, sau đó quay ngẫu nhiên quanh tâm hình.
        h = int(radius_or_half * 1.65)
        points = np.array([[0, -2 * h / 3], [-radius_or_half, h / 3],
                           [radius_or_half, h / 3]], dtype=np.float32)
        points = _rotate_points(points, (0, 0), angle) + np.asarray(center, dtype=np.int32)
        cv2.fillPoly(image, [points], color, lineType=cv2.LINE_AA)

    # Nhiễu nhẹ giúp dữ liệu bớt hoàn hảo nhưng vẫn giữ tương phản rõ.
    noise = rng.normal(0, 2.0, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def generate_dataset(dataset_dir: Path, images_per_class: int = 50, seed: int = 42) -> None:
    """Sinh và ghi ảnh thành dataset_dir/<class>/<class>_NNN.png."""
    rng = np.random.default_rng(seed)
    for label in CLASSES:
        class_dir = dataset_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(images_per_class):
            image = make_shape_image(label, rng)
            destination = class_dir / f"{label}_{index + 1:03d}.png"
            if not cv2.imwrite(str(destination), image):
                raise OSError(f"Không ghi được ảnh: {destination}")


def preprocess_image(gray: np.ndarray) -> Dict[str, np.ndarray | float]:
    """Grayscale -> Gaussian Blur -> Otsu -> Canny -> contour lớn nhất."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_value, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Bỏ các contour rất nhỏ có thể do nhiễu; chọn vật thể có diện tích lớn nhất.
    contours = [c for c in contours if cv2.contourArea(c) >= 20]
    contour = max(contours, key=cv2.contourArea) if contours else None
    return {"gray": gray, "blurred": blurred, "threshold": binary, "edges": edges,
            "contour": contour, "threshold_value": float(threshold_value)}


def extract_shape_features(processed: Dict[str, np.ndarray | float]) -> np.ndarray:
    """6 giá trị: A, P, AR, circularity, edge density, solidity.

    Đề bài nêu 5 đại lượng nhưng yêu cầu vector 6 giá trị; solidity (A / diện
    tích convex hull) được thêm làm đặc trưng hình dạng thứ sáu.
    """
    contour = processed["contour"]
    if contour is None:
        return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float32)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = width / height if height else 0.0
    circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter else 0.0

    mask = np.zeros_like(processed["threshold"])
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    object_pixels = max(int(np.count_nonzero(mask)), 1)
    edge_pixels = int(np.count_nonzero((processed["edges"] > 0) & (mask > 0)))
    edge_density = edge_pixels / object_pixels
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area else 0.0
    return np.asarray([area, perimeter, aspect_ratio, circularity, edge_density, solidity], dtype=np.float32)


def extract_hog_features(processed: Dict[str, np.ndarray | float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả descriptor HOG, ảnh 64x64 và ảnh HOG để trực quan hóa."""
    gray = processed["gray"]
    normalized = cv2.resize(gray, HOG_SIZE, interpolation=cv2.INTER_AREA)
    descriptor = hog(normalized, visualize=True, **HOG_PARAMS)
    # skimage trả về (descriptor, hog_image) khi visualize=True.
    descriptor, hog_image = descriptor
    return descriptor.astype(np.float32), normalized, hog_image.astype(np.float32)


def visualize_processing(image: np.ndarray, output_file: Path | None = None, show: bool = True) -> None:
    """Hiển thị ảnh gốc/xám, blur, Otsu, Canny và contour lớn nhất."""
    processed = preprocess_image(image)
    contour_view = cv2.cvtColor(processed["gray"], cv2.COLOR_GRAY2RGB)
    if processed["contour"] is not None:
        cv2.drawContours(contour_view, [processed["contour"]], -1, (255, 0, 0), 2)
    original = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    panels = [("Ảnh đầu vào", original), ("Grayscale", processed["gray"]),
              ("Gaussian Blur", processed["blurred"]), ("Otsu Threshold", processed["threshold"]),
              ("Canny", processed["edges"]), ("Contour lớn nhất", contour_view)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for axis, (title, panel) in zip(axes.ravel(), panels):
        axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def visualize_hog(image: np.ndarray, output_file: Path | None = None, show: bool = True) -> None:
    processed = preprocess_image(image)
    _, normalized, hog_image = extract_hog_features(processed)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(normalized, cmap="gray")
    axes[0].set_title("Ảnh chuẩn hóa 64×64")
    axes[1].imshow(hog_image, cmap="gray")
    axes[1].set_title("Trực quan HOG")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def load_feature_table(dataset_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows: List[dict] = []
    images: Dict[str, np.ndarray] = {}
    for label in CLASSES:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Thiếu thư mục lớp {label}: {class_dir}")
        for image_path in sorted(class_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Cảnh báo: bỏ qua ảnh lỗi {image_path}")
                continue
            processed = preprocess_image(image)
            shape_vector = extract_shape_features(processed)
            hog_vector, _, _ = extract_hog_features(processed)
            row = {name: float(value) for name, value in zip(SHAPE_FEATURE_NAMES, shape_vector)}
            row.update({f"hog_{index:04d}": float(value) for index, value in enumerate(hog_vector)})
            row.update({"label": label, "file": str(image_path)})
            rows.append(row)
            images[str(image_path)] = image
    if not rows:
        raise ValueError(f"Không có ảnh PNG hợp lệ trong {dataset_dir}")
    return pd.DataFrame(rows), images


def train_and_evaluate(frame: pd.DataFrame, random_state: int = 42):
    shape_columns = SHAPE_FEATURE_NAMES
    hog_columns = [column for column in frame.columns if column.startswith("hog_")]
    scenarios = {
        "Shape": shape_columns,
        "HOG": hog_columns,
        "Shape + HOG": shape_columns + hog_columns,
    }
    model_factories = {
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
        "SVM": lambda: SVC(kernel="rbf", C=10.0, gamma="scale"),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=random_state,
                                                          class_weight="balanced"),
    }
    indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=random_state, stratify=frame["label"]
    )
    y = frame["label"].to_numpy()
    class_order = list(CLASSES)
    models = {}
    metrics_rows = []
    confusion_matrices = {}
    predictions = {}
    for scenario, columns in scenarios.items():
        x_train = frame.iloc[train_indices][columns].to_numpy(dtype=np.float32)
        x_test = frame.iloc[test_indices][columns].to_numpy(dtype=np.float32)
        for model_name, factory in model_factories.items():
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", factory())])
            pipeline.fit(x_train, y[train_indices])
            prediction = pipeline.predict(x_test)
            key = (scenario, model_name)
            models[key] = pipeline
            predictions[key] = prediction
            metrics_rows.append({"Scenario": scenario, "Model": model_name,
                                 "Accuracy": accuracy_score(y[test_indices], prediction),
                                 "F1_macro": f1_score(y[test_indices], prediction, average="macro")})
            confusion_matrices[key] = confusion_matrix(y[test_indices], prediction, labels=class_order)
    return pd.DataFrame(metrics_rows), models, confusion_matrices, predictions, test_indices, class_order


def save_confusion_matrices(matrices: dict, class_order: List[str], output_file: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for axis, (key, matrix) in zip(axes.ravel(), matrices.items()):
        axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(class_order)), [CLASS_NAMES_VI[x] for x in class_order], rotation=25, ha="right")
        axis.set_yticks(range(len(class_order)), [CLASS_NAMES_VI[x] for x in class_order])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_title(f"{key[0]} / {key[1]}")
        axis.set_xlabel("Dự đoán")
        axis.set_ylabel("Thực tế")
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_new_image_prediction(models: dict, rng: np.random.Generator, output_dir: Path) -> dict:
    """Sinh một mẫu mới không thuộc tập train/test và dự đoán bằng 9 pipeline."""
    true_label = str(rng.choice(CLASSES))
    image = make_shape_image(true_label, rng)
    processed = preprocess_image(image)
    shape_vector = extract_shape_features(processed)
    hog_vector, _, hog_image = extract_hog_features(processed)
    predictions = {}
    for (scenario, model_name), model in models.items():
        if scenario == "Shape":
            vector = shape_vector
        elif scenario == "HOG":
            vector = hog_vector
        else:
            vector = np.concatenate([shape_vector, hog_vector])
        predictions[f"{scenario} / {model_name}"] = str(model.predict(vector.reshape(1, -1))[0])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Ảnh mới — nhãn thật: {CLASS_NAMES_VI[true_label]}")
    axes[0].axis("off")
    axes[1].imshow(hog_image, cmap="gray")
    axes[1].set_title("Đặc trưng HOG")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "du_doan_anh_moi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"true_label": true_label, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân loại hình tròn/vuông/tam giác bằng Shape và HOG.")
    parser.add_argument("--output-dir", type=Path, default=Path("ket_qua_shape_hog"))
    parser.add_argument("--images-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true", help="Lưu biểu đồ nhưng không mở cửa sổ.")
    args = parser.parse_args()
    if args.images_per_class < 5:
        parser.error("--images-per-class cần ít nhất 5 ảnh mỗi lớp để chia train/test.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.output_dir / "dataset_shapes"
    generate_dataset(dataset_dir, args.images_per_class, args.seed)
    frame, images = load_feature_table(dataset_dir)
    feature_columns = SHAPE_FEATURE_NAMES + [c for c in frame.columns if c.startswith("hog_")]
    frame.to_csv(args.output_dir / "feature_table.csv", index=False, encoding="utf-8-sig")
    print(f"Đã tạo và trích xuất {len(frame)} ảnh. Số chiều: {len(SHAPE_FEATURE_NAMES)} Shape + "
          f"{len(feature_columns) - len(SHAPE_FEATURE_NAMES)} HOG.")
    print("Số ảnh mỗi lớp:\n" + frame["label"].value_counts().sort_index().to_string())

    example_path = frame.iloc[0]["file"]
    example_image = images[example_path]
    visualize_processing(example_image, args.output_dir / "preprocessing_steps.png", show=not args.no_show)
    visualize_hog(example_image, args.output_dir / "hog_visualization.png", show=not args.no_show)

    try:
        metrics, models, matrices, _, _, class_order = train_and_evaluate(frame, args.seed)
    except ValueError as error:
        raise SystemExit(f"Không thể train/evaluate: {error}") from error
    metrics.to_csv(args.output_dir / "metrics_9_cases.csv", index=False, encoding="utf-8-sig")
    save_confusion_matrices(matrices, class_order, args.output_dir / "confusion_matrices_9_cases.png")
    new_prediction = save_new_image_prediction(models, np.random.default_rng(args.seed + 1), args.output_dir)
    write_report(metrics, new_prediction, args.output_dir / "report.md")

    print("\n=== KẾT QUẢ 9 TRƯỜNG HỢP ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nNhãn thật ảnh mới: {CLASS_NAMES_VI[new_prediction['true_label']]}")
    for key, value in new_prediction["predictions"].items():
        print(f"{key}: {CLASS_NAMES_VI[value]}")
    print(f"\nĐã lưu toàn bộ sản phẩm tại: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
