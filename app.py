"""Command-line interface for the Plant Disease Detection system."""

from predict import load_model, predict


def main() -> None:
    print("Plant Disease Detection AI")
    model, class_names = load_model()

    while True:
        image_path = (
            input("\nEnter the path to a leaf image (or 'q' to quit): ")
        .strip()
        .strip('"')
        )

        if image_path.lower() == "q":
            break

        if not image_path:
            continue

        try:
            label, confidence = predict(image_path, model=model, class_names=class_names)
        except FileNotFoundError:
            print(f"Could not find image: {image_path}")
            continue
        except Exception as e:
            print(f"Prediction Failed: {e}")
            continue
        print(f"Predicted disease: {label}")
        print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
    
