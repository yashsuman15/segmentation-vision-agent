import gradio as gr
from io_utils import load_image
from grounded_sam import detect, segment
from plot_utils import annotate, overlay_masks
from PIL import Image
import numpy as np

def inference(image, object_names, mask_only, show_counts):
    # Normalize input labels (case-insensitive, strip whitespace)
    labels = [label.strip().lower() for label in object_names.split(",") if label.strip()]
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Run detection and segmentation
    detections = detect(image, labels)
    detections = segment(image, detections)
    
    # Prepare visualization
    if mask_only:
        # Create mask overlay with random colors per instance
        result_img = overlay_masks(image, detections, alpha=0.5)
    else:
        # Show annotated image with boxes/labels
        result_img = annotate(image, detections)
    
    # Calculate object counts
    counts_str = ""
    if show_counts:
        counts = {}
        # Count all detected objects (case-insensitive)
        for det in detections:
            label = det.label.strip().lower()
            counts[label] = counts.get(label, 0) + 1
        
        # Format counts string
        if counts:
            counts_str = "Detected Objects:\n" + "\n".join(
                [f"- {label}: {count}" for label, count in counts.items()]
            )
        else:
            counts_str = "No objects detected."
    
    return result_img, counts_str




demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Textbox(label="Object Name(s), comma separated (e.g., cat, dog, car)"),
        gr.Checkbox(label="Show Mask Only", value=False),
        gr.Checkbox(label="Show object counts per class", value=False)
    ],
    outputs=[
        gr.Image(type="numpy", label="Result"),
        gr.Textbox(label="Object Counts")
    ],
    title="Segment Anything with Object Name(s)",
    description="Upload an image and enter object names, comma separated. Toggle options to view masks or object counts."
)

if __name__ == "__main__":
    demo.launch(pwa=True)
