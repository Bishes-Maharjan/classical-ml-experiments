import numpy as np
import cv2
import glob
import os
import re

# Folders
main_folder = "unstiched-images"
output_folder = "stitched-output-advanced"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def natural_sort(s):
    """Sort files naturally (1, 2, 10 instead of 1, 10, 2)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def stitch_two_images(img1, img2):
    """Stitch two images using feature matching"""
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if abs(w1 - w2) < abs(h1 - h2):
        direction = 'vertical'
    else:
        direction = 'horizontal'
    
    # Extract overlap regions (search 20% of image)
    if direction == 'vertical':
        overlap = int(h1 * 0.2)
        roi1 = img1[-overlap:, :]  # Bottom of img1
        roi2 = img2[:overlap, :]    # Top of img2
        offset1_y = h1 - overlap
        offset1_x = 0
    else:
        overlap = int(w1 * 0.2)
        roi1 = img1[:, -overlap:]   # Right of img1
        roi2 = img2[:, :overlap]     # Left of img2
        offset1_x = w1 - overlap
        offset1_y = 0
    
    # Find features
  
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(roi1, None)
    kp2, des2 = sift.detectAndCompute(roi2, None)
        
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)

    
    # Filter good matches
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good.append(m)
    
    print(f"  Direction: {direction}, Found {len(good)} matches")
    
    # Need at least 10 matches
    if len(good) < 10:
        raise Exception(f"Not enough matches! Only found {len(good)} matches. Images might not overlap enough or lack distinctive features.")
    
    # Adjust keypoint positions to full image coordinates
    for kp in kp1:
        kp.pt = (kp.pt[0] + offset1_x, kp.pt[1] + offset1_y)
    
    # Get matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    
    if H is None:
        raise Exception("Homography calculation failed! Images are too different or matches are poor quality.")
    
    # Calculate output size
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, H)
    
    all_corners = np.concatenate([
        np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
        corners2_transformed
    ], axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Warp and combine
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    output_size = (x_max - x_min, y_max - y_min)
    
    warped2 = cv2.warpPerspective(img2, translation.dot(H), output_size)
    result = warped2.copy()
    result[-y_min:h1-y_min, -x_min:w1-x_min] = img1
    
    return result

def stitch_multiple(images):
    """Stitch multiple images sequentially"""
    result = images[0]
    for i in range(1, len(images)):
        print(f"  Stitching image {i+1}/{len(images)}")
        result = stitch_two_images(result, images[i])
    return result

# Process each folder
for folder_num in range(1, 6):
    folder_path = os.path.join(main_folder, str(folder_num))
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.*")), key=natural_sort)
    
    print(f"\n{'='*60}")
    print(f"Processing folder {folder_num}: {len(image_paths)} images found")
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            print(f"  Loaded: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")
    
    if len(images) < 2:
        print(f"ERROR: Need at least 2 images, found {len(images)}")
        continue
    
    try:
        stitched = stitch_multiple(images)
        output_path = os.path.join(output_folder, f"stitched_{folder_num}.png")
        cv2.imwrite(output_path, stitched)
        print(f"✓ SUCCESS: {output_path} ({stitched.shape[1]}x{stitched.shape[0]})")
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")

print(f"\n{'='*60}")
print("Done!")