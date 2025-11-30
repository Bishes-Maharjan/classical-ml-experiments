# Image Stitching Project

This project automatically stitches multiple images from folders to create panoramic-like outputs. It includes both a simple OpenCV-based approach and an advanced feature-based approach using SIFT and FLANN.

## Features

• Automatic stitching of multiple images per folder
• Two approaches: 1. Simple Stitching: OpenCV built-in Stitcher 2. Advanced Stitching: Custom SIFT + FLANN + homography
• Natural sorting of images for correct order
• Handles horizontal and vertical stitching
• Robust error handling for insufficient matches or invalid images
• Saves outputs in separate folders

⸻

## Folder Structure

project/
├─ unstiched-images/ # Input images organized in subfolders
│ ├─ 1/
│ ├─ 2/
│ └─ ...
├─ stitched-output/ # Output for simple OpenCV stitching
├─ stitched-output-advanced/ # Output for advanced feature-based stitching
├─ stitch_simple.py # Simple OpenCV stitching script
├─ stitch_advanced.py # Advanced SIFT + FLANN stitching script
└─ README.md

⸻

## Requirements

• Python 3.8+
• OpenCV (opencv-python)
• imutils
• NumPy

Install dependencies:

```bash
pip install opencv-python imutils numpy
```

⸻

## Usage

Simple Stitching

python stitch_simple.py

    •	Reads images from unstiched-images/
    •	Resizes images for faster processing
    •	Saves stitched results in stitched-output/

Advanced Stitching

python stitch_advanced.py

    •	Reads images from unstiched-images/
    •	Performs natural sorting of filenames
    •	Uses SIFT for feature detection and FLANN for matching
    •	Calculates homography and warps images
    •	Handles horizontal and vertical stitching
    •	Saves stitched results in stitched-output-advanced/

⸻

## Notes

• Each folder must contain at least 2 images, that have overlap between them
• Advanced stitching may fail if there are insufficient feature matches
• Advanced stitching provides better results for uneven overlap or orientation

⸻

## Output

• stitched-output/_<folder_number>.png → Simple OpenCV stitching
• stitched-output-advanced/_<folder_number>.png → Advanced feature-based stitching
