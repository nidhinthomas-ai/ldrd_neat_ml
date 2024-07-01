"""
The purpose of this module is to replicate key
parts of the main image analysis workflow, but
apply them only to a single plate reader image
for faster iteration/improvement of bubble
detection/analysis methods.

Sample incantation:
time python neat_ml/misc/single_image_iterate.py --image-path '/home/treddy/LANL/LDRD_DR_NEAT_data/Images/DEXTRAN (10k) 2~14wt_ (with PEO 10K)/DEXTRAN 12wt_/DEX12wt_,PEO10wt_.tiff'
"""

import os
import argparse
import time

from neat_ml.lib import run_cmd

import cv2
import skimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def opencv_blob_detection_single_image(image_path, debug: bool = False):
    start = time.perf_counter()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 1_000_000
    params.minThreshold = 1
    params.maxThreshold = 3000
    params.thresholdStep = 1
    params.minConvexity = 0.89
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params) # type: ignore[attr-defined]
    # actual detection of blobs happens:
    keypoints = detector.detect(image)
    num_blobs_img = len(keypoints)
    end = time.perf_counter()
    exe_time_sec = end - start
    if debug:
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        image_orig = skimage.color.gray2rgb(image)
        axs[0].imshow(image_orig)
        axs[0].set_title("original")
        blob_image = cv2.drawKeypoints(image.copy(), #type: ignore[call-overload]
                                       keypoints,
                                       None,
                                       (255, 0, 0),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(blob_image,
                       (int(x), int(y)),
                       color=(255, 0, 0),
                       radius=int(kp.size/2),
                       thickness=-1)
        axs[1].imshow(blob_image)
        axs[1].set_title(f"OpenCV SimpleBlobDetector (Found {num_blobs_img} blobs in {exe_time_sec:.2f} s)")
        fig.tight_layout()
        fig.savefig(f"OpenCV_blob_detection_debug.png",
                    dpi=300,
                    pad_inches=0.1,
                    bbox_inches='tight')
        matplotlib.pyplot.close()


def kim_park_blob_detect_single_image(image_path, debug: bool = False):
    start = time.perf_counter()
    # need an RGB JPG for their DL code...
    jpg_version = image_path.replace("tiff", "jpg")
    # also, easiest to place each image in its own
    # dir for Kim and Park code...
    jpg_version_fname = os.path.basename(jpg_version)
    new_dir = os.path.join("/tmp/", f"{jpg_version_fname}"[:-4])
    os.makedirs(new_dir, exist_ok=True)
    new_jpg_filepath = os.path.join(new_dir, jpg_version_fname)
    new_jpg_dir = os.path.dirname(new_jpg_filepath)
    run_cmd(f"convert '{image_path}' '{new_jpg_filepath}'")
    bub_path = os.path.join("/home/treddy/github_projects/BubMask")
    weights_path = os.path.join(bub_path, "mask_rcnn_bubble.h5")
    bubble_path = os.path.join(bub_path, "bubble/bubble.py")
    results_path = os.path.join(os.getcwd(), "debug_deep_learning_bub_results")
    os.makedirs(results_path, exist_ok=True)
    py_36 = "/home/treddy/miniforge3/envs/py_36_neat/bin/python"
    run_cmd(f"{py_36} {bubble_path} detect --weights={weights_path} --image='{new_jpg_dir}' --results={results_path} --confidence=0.5")
    base_str = jpg_version_fname[:-4]
    # this path is ugly, though I believe partly determined
    # by the unmaintained DL bubble code
    results_filepath = os.path.join(results_path,
                                    base_str,
                                    base_str,
                                    base_str + ".txt")
    blob_counter = 0
    blob_data = pd.read_csv(results_filepath)
    # the column names have whitespaces and quotes I want cleaned up
    clean_col_names = ['x', 'y', 'Orientation', 'Axis_major_length', 'Axis_minor_length', 'Area']
    blob_data.columns = clean_col_names
    print(blob_data)
    num_blobs_detected = blob_data.shape[0]
    median_blob_area = np.median(blob_data["Area"])
    end = time.perf_counter()
    exe_time_sec = end - start
    if debug:
        print("Kim and Park num_blobs_detected:", num_blobs_detected)
        print("Kim and Park median_blob_area:", median_blob_area)
        print(f"Kim and Park single image execution time: {exe_time_sec:.3f} seconds")
        # produce the side-by-side original image vs. DL-bubble-labeled image
        # using OpenCV to "color" the bubbles
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_orig = skimage.color.gray2rgb(image)
        axs[0].imshow(image_orig)
        axs[0].set_title("original")
        label_image = image_orig.copy()
        for circle in tqdm(blob_data.itertuples(),
                           desc="Drawing Kim and Park blob labels"):
            cv2.circle(label_image,
                       (int(circle.x), int(circle.y)),
                       int(circle.Axis_major_length / 2),
                       (255, 0, 0),
                       -1,
                       )
        axs[1].imshow(label_image)
        axs[1].set_title(f"Kim and Park DL (Found {num_blobs_detected} blobs in {exe_time_sec:.2f} s)")
        fig.tight_layout()
        fig.savefig(f"kim_park_blob_detection_debug.png",
                    dpi=300,
                    pad_inches=0.1,
                    bbox_inches='tight')
        matplotlib.pyplot.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="Path to the plate reader image file to be processed")
    parser.add_argument("--blob-method", type=str, help="Blob detection method to use")
    args = parser.parse_args()
    if args.blob_method == "opencv":
        opencv_blob_detection_single_image(image_path=args.image_path,
                                           debug=True)
    elif args.blob_method == "kim_park":
        kim_park_blob_detect_single_image(image_path=args.image_path,
                                          debug=True)
    else:
        raise ValueError(f"Unsupported blob detection method: {args.blob_method}")
