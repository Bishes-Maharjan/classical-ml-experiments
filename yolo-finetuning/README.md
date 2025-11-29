# Fine-Tuned YOLO Models

This repository contains fine-tuned YOLO (You Only Look Once) object detection models for custom object detection tasks.

## Models

This repository includes two fine-tuned YOLO model versions:

- **YOLOv5** - Ultralytics YOLOv5 model fine-tuned on custom dataset
- **YOLOv8** - Ultralytics YOLOv8 model fine-tuned on custom dataset

Both models are trained for the same detection task and can be used interchangeably depending on your performance and accuracy requirements.

## Directory Structure

```
.
├── yolov5/
│   ├── weights/
│   │   └── best.pt
│   └── ...
├── yolov8/
│   ├── weights/
│   │   └── best.pt
│   └── ...
└── README.md
```