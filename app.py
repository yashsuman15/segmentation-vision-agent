import gradio as gr
from io_utils import load_image
from grounded_sam import detect, segment
from plot_utils import annotate, overlay_masks
from PIL import Image
import numpy as np

def inference(image, object_names, mask_only):
    labels = [label.strip() for label in object_names.split(",") if label.strip()]
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    detections = detect(image, labels)
    detections = segment(image, detections)
    if mask_only:
        result = overlay_masks(image, detections, alpha=0.5)
    else:
        result = annotate(image, detections)
    return result

demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Textbox(label="Object Name(s), comma separated (e.g., cat, dog, car)"),
        gr.Checkbox(label="Show Mask Only", value=True)
    ],
    outputs=gr.Image(type="numpy", label="Result"),
    title="Segmentation Vision Agent",
    description="Upload an image and enter object names, comma separated. Toggle 'Show Mask Only' to view only the colored segmentation masks."
)

if __name__ == "__main__":
    demo.launch(pwa=True)
