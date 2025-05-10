# main.py

from io_utils import load_image
from grounded_sam import detect, segment
from plot_utils import plot_detections

def main():
    # 1. Load image (local path or URL)
    image_path = "sample.jpg"  # Replace with your image path or URL
    image = load_image(image_path)

    # 2. Define labels to detect
    labels = ["cat", "dog"]  # Replace with your target labels

    # 3. Run detection
    detections = detect(image, labels)

    # 4. Run segmentation
    detections = segment(image, detections)

    # 5. Visualize results
    plot_detections(image, detections)

if __name__ == "__main__":
    main()
