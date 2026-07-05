"""Exploratory analysis of the PlantVillage dataset.

Inspects class structure, image counts, class balance, image
dimensions, and sample images before any preprocessing or training.
"""

import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "PlantVillage"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
SAMPLES_PER_DIMENSION_CHECK = 20
SAMPLE_IMAGES_TO_DISPLAY = 9


def get_class_names(split_dir: Path) -> list[str]:
    return sorted(d.name for d in split_dir.iterdir() if d.is_dir())


def count_images_per_class(split_dir: Path, class_names: list[str]) -> Counter:
    counts = Counter()
    for class_name in class_names:
        class_dir = split_dir / class_name
        counts[class_name] = sum(1 for f in class_dir.iterdir() if f.is_file())
    return counts


def print_class_summary(split_name: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"\n{split_name} set: {len(counts)} classes, {total} images")
    for class_name, count in counts.most_common():
        print(f"  {class_name:<40} {count}")

    if counts:
        largest = max(counts.values())
        smallest = min(counts.values())
        imbalance_ratio = largest / smallest if smallest else float("inf")
        print(f"  Imbalance ratio (largest/smallest class): {imbalance_ratio:.2f}")


def plot_class_distribution(counts: Counter, split_name: str) -> None:
    classes = list(counts.keys())
    values = [counts[c] for c in classes]

    plt.figure(figsize=(12, 6))
    plt.bar(classes, values)
    plt.xticks(rotation=90)
    plt.ylabel("Number of images")
    plt.title(f"Class distribution — {split_name}")
    plt.tight_layout()
    plt.show()


def check_image_dimensions(split_dir: Path, class_names: list[str]) -> None:
    dimensions = Counter()
    modes = Counter()

    for class_name in class_names:
        class_dir = split_dir / class_name
        image_paths = list(class_dir.iterdir())[:SAMPLES_PER_DIMENSION_CHECK]
        for image_path in image_paths:
            with Image.open(image_path) as img:
                dimensions[img.size] += 1
                modes[img.mode] += 1

    print("\nImage dimensions (sampled):")
    for size, count in dimensions.most_common():
        print(f"  {size}: {count}")

    print("Image color modes (sampled):")
    for mode, count in modes.most_common():
        print(f"  {mode}: {count}")


def show_sample_images(split_dir: Path, class_names: list[str]) -> None:
    sample_classes = random.sample(
        class_names, min(SAMPLE_IMAGES_TO_DISPLAY, len(class_names))
    )

    plt.figure(figsize=(12, 12))
    for i, class_name in enumerate(sample_classes):
        class_dir = split_dir / class_name
        image_path = random.choice(list(class_dir.iterdir()))
        img = Image.open(image_path)

        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(class_name, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def explore_split(split_dir: Path, split_name: str) -> None:
    class_names = get_class_names(split_dir)
    counts = count_images_per_class(split_dir, class_names)
    print_class_summary(split_name, counts)
    plot_class_distribution(counts, split_name)
    check_image_dimensions(split_dir, class_names)
    show_sample_images(split_dir, class_names)


def main() -> None:
    explore_split(TRAIN_DIR, "train")
    explore_split(VAL_DIR, "val")


if __name__ == "__main__":
    main()
