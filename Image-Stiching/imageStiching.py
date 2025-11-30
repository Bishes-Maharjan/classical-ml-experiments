import numpy as np
import cv2
import glob
import os
import imutils

main_folder = "unstiched-images"
output_folder = "stitched-output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for subfolder in range(1, 6):
    folder_path = os.path.join(main_folder, str(subfolder))
    image_paths = glob.glob(os.path.join(folder_path, "*.*"))
    images = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            img = imutils.resize(img, width=300)
            images.append(img)


    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        output_path = os.path.join(output_folder, f"stitched_{subfolder}.png")
        cv2.imwrite(output_path, stitched_image)
        print(f" Stitched folder {subfolder} → {output_path}")
    else:
        print(f" Stitching failed for folder {subfolder}, status code: {status}")