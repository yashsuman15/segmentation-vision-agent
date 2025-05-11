import gradio as gr
from io_utils import load_image
from grounded_sam import detect, segment
from plot_utils import annotate, overlay_masks
from PIL import Image
import numpy as np
import torch

# Define a global cache to store results (for demo purposes; use proper state management in production)
result_cache = {}

def inference(image, object_names):
    # Clear previous cache
    result_cache.clear()
    
    # Process image
    labels = [label.strip().lower() for label in object_names.split(",") if label.strip()]
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Run detection and segmentation
    detections = detect(pil_image, labels)
    detections = segment(pil_image, detections)
    
    # Generate both versions of the image
    annotated_img = annotate(pil_image, detections)
    mask_img = overlay_masks(pil_image, detections)
    
    # Count objects
    counts = {}
    for det in detections:
        label = det.label.strip().lower()
        counts[label] = counts.get(label, 0) + 1
    
    # Store results in cache
    result_cache.update({
        "original": pil_image,
        "annotated": annotated_img,
        "mask": mask_img,
        "counts": counts,
        "detections": detections
    })
    
    # Return initial view (annotated image) and control visibility
    return annotated_img, gr.Checkbox(visible=True), gr.Checkbox(visible=True)

def toggle_mask(show_mask):
    if show_mask and "mask" in result_cache:
        return result_cache["mask"]
    return result_cache.get("annotated", result_cache.get("original"))

def toggle_counts(show_counts):
    if show_counts and "counts" in result_cache:
        counts = result_cache["counts"]
        count_str = "\n".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "No objects detected"
        return gr.update(value=count_str, visible=True)
    else:
        return gr.update(value="", visible=False)


with gr.Blocks() as demo:
    gr.Markdown("# Object Segmentation Analyzer")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image", type="numpy")
            text_input = gr.Textbox(label="Object Names (comma separated)")
            submit_btn = gr.Button("Process Image")
        
        with gr.Column():
            output_image = gr.Image(label="Result", interactive=False)
            count_output = gr.Textbox(label="Object Counts", visible=False)
            with gr.Row():
                mask_toggle = gr.Checkbox(label="Show Masks", visible=False)
                count_toggle = gr.Checkbox(label="Show Counts", visible=False)

    submit_btn.click(
        fn=inference,
        inputs=[image_input, text_input],
        outputs=[output_image, mask_toggle, count_toggle]
    )
    
    mask_toggle.change(
        fn=toggle_mask,
        inputs=mask_toggle,
        outputs=output_image
    )
    
    count_toggle.change(
        fn=toggle_counts,
        inputs=count_toggle,
        outputs=count_output
    )


if __name__ == "__main__":
    demo.launch()

