import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Union
from PIL import Image
from data_structures import DetectionResult

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

# Add plotly functions as needed
def overlay_masks(image, detections, alpha=0.5, outline_thickness=3):
    """
    Overlay segmentation masks on the image with random colors and transparency.
    Draw a solid outline for each mask in the same color as the mask.
    """
    image = np.array(image).copy()
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    overlay = image.copy()
    rng = np.random.default_rng()  # True randomness

    for det in detections:
        mask = det.mask
        if mask is not None:
            # Generate a random color for this mask
            color = rng.integers(0, 256, size=3).tolist()

            # Apply colored mask with transparency
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (alpha * np.array(color) + (1 - alpha) * overlay[mask_bool]).astype(np.uint8)

            # Find contours and draw outline in the same color
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, thickness=outline_thickness)

    return overlay

