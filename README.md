# Segmentation Vision Agent

**Segmentation Vision Agent** is a modular Python project for object detection and segmentation in images, featuring an interactive Gradio web interface.

## Key Features

- **Input:** Accepts an image and one or more object names (comma-separated).
- **Detection:** Uses a Grounded DINO-based detector to find objects matching the provided names.
- **Segmentation:** Applies Segment Anything Model (SAM) to generate segmentation masks for detected objects.
- **Visualization:** 
  - Annotates images with bounding boxes, labels, and outlines.
  - Optionally overlays colored, semi-transparent masks for each detected object.
- **Web App:** Provides a Gradio-based interface for easy user interaction, including toggling between annotation and mask-only views.

## Project Structure

- `data_structures.py`: Defines core data classes (`BoundingBox`, `DetectionResult`).
- `mask_utils.py`: Functions for mask/polygon conversion and mask refinement.
- `plot_utils.py`: Visualization utilities for annotation and mask overlays.
- `io_utils.py`: Image loading and bounding box extraction.
- `grounded_sam.py`: Detection and segmentation logic using pretrained models.
- `main.py`: Script for running the pipeline from the command line.
- `gradio_app.py`: Launches the Gradio web interface for interactive use.

## Usage

- **Command-line:**  
  Run the pipeline with:
    
    `python main.py`

- **Web Interface:**  
Launch the interactive app with:
    
    `python gradio_app.py`


## Dependencies

- `torch`
- `transformers`
- `opencv-python`
- `matplotlib`
- `pillow`
- `plotly`
- `gradio`

Dependencies are managed via `pyproject.toml`. Use the following command to add packages and update your project configuration:

## Example Workflow

1. Upload an image and enter object names (e.g., "cat, dog").
2. The system detects and segments all matching objects.
3. The result is displayed with either annotated outlines/labels or colored masks, based on user selection.

---

**Summary:**  
This project provides a modular, extensible framework for vision-based object detection and segmentation, with both scriptable and interactive web-based interfaces, and clear separation of concerns for easy maintenance and extension.

---
## Project Structure

vision_agent/
├── __init__.py
├── data_structures.py       # BoundingBox, DetectionResult
├── plot_utils.py            # annotate, plot_detections, plot_detections_plotly, etc.
├── mask_utils.py            # mask_to_polygon, polygon_to_mask, refine_masks
├── io_utils.py              # load_image, get_boxes
├── grounded_sam.py          # detect, segment
└── main.py                  # entry point for your application

---

## Project Workflow

Below is a visual representation of how our image segmentation system works:

```mermaid
flowchart TD
    A[User] -->|Uploads image & \nprovides object names| B[Input Processing]
    B --> C[Object Detection \nusing Grounded DINO]
    C -->|Detected boxes \n& labels| D[Object Segmentation \nusing SAM]
    D -->|Segmentation masks| E{Visualization \nChoice}
    E -->|Option 1| F[Annotated Outlines/Boxes]
    E -->|Option 2| G[Mask-only Overlay]
    F --> H[Result Display]
    G --> H
    H --> A
    
    classDef userAction fill:#d4f1f9,stroke:#05a,stroke-width:2px
    classDef processing fill:#ffe6cc,stroke:#d79b00,stroke-width:1px
    classDef models fill:#d5e8d4,stroke:#82b366,stroke-width:1px
    classDef choice fill:#fff2cc,stroke:#d6b656,stroke-width:1px
    classDef output fill:#f8cecc,stroke:#b85450,stroke-width:1px
    
    class A userAction
    class B,H processing
    class C,D models
    class E choice
    class F,G output