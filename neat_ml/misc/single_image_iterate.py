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
from scipy.spatial.distance import cdist


def filter_close_contacts(df, pixel_dist=2):
    # TODO: check if this func makes sense for more than
    # 2 zoom values combined?
    coords = df.iloc[:, :2]
    dist = cdist(coords, coords)
    # filter as duplicates bubbles within pixel_dist pixels
    close_dist_row_indices, close_dist_col_indices = np.nonzero(dist < pixel_dist)
    # ignore self matches of course
    mask = (close_dist_row_indices != close_dist_col_indices)
    close_dist_row_indices = close_dist_row_indices[mask]
    close_dist_col_indices = close_dist_col_indices[mask]
    if not np.array_equal(np.sort(close_dist_row_indices), np.sort(close_dist_col_indices)):
        raise ValueError("symmetric dist matrix is not symmetric on indices")
    # should be safe to filter out one member of the close pair
    # and keep the other member
    keep_rows = set()
    exclude_rows = set()
    for row1, row2 in zip(close_dist_row_indices, close_dist_col_indices):
        if row1 not in exclude_rows:
            keep_rows.add(row1)
        if row2 not in keep_rows:
            exclude_rows.add(row2)
    # should halve the number of close match rows:
    assert len(keep_rows) + len(exclude_rows) == close_dist_row_indices.size
    close_rows_to_keep = list(keep_rows)
    # we also want to keep any rows that are not in the exclude data
    # nor in the close contact keep data (the unique bubbles with no close contacts)
    unique_close_indices = np.unique(close_dist_row_indices)
    far_rows_to_keep = df.index[~np.isin(df.index, unique_close_indices)]
    rows_to_keep = list(close_rows_to_keep) + list(far_rows_to_keep)
    rows_to_keep = np.sort(rows_to_keep)

    # we also want to retain the rows that did not have
    # close contacts
    boolean_indexer = np.isin(df.index, rows_to_keep)
    df = df[boolean_indexer]
    return df


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
    base_str = jpg_version_fname[:-4]
    # this path is ugly, though I believe partly determined
    # by the unmaintained DL bubble code
    results_filepath = os.path.join(results_path,
                                    base_str,
                                    base_str,
                                    base_str + ".txt")
    clean_col_names = ['x', 'y', 'Orientation', 'Axis_major_length', 'Axis_minor_length', 'Area']
    # scan over a range of "zoom" values and aggregate
    # the results (Kim and Park recommend adjusting zoom
    # to detect bubbles across a range of sizes)
    orig_jpg = cv2.imread(new_jpg_filepath)
    list_zoom_dfs = []
    for zoom in [1.0, 1.5]:
        zoom_in_jpg = cv2.resize(orig_jpg,
                                 None,
                                 fx=zoom,
                                 fy=zoom,
                                 interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(new_jpg_filepath, zoom_in_jpg)
        run_cmd(f"{py_36} {bubble_path} detect --weights={weights_path} --image='{new_jpg_dir}' --results={results_path} --confidence=0.5")
        blob_data = pd.read_csv(results_filepath)
        # the column names have whitespaces and quotes I want cleaned up
        blob_data.columns = clean_col_names
        # compensate for zoom:
        blob_data.x /= zoom
        blob_data.y /= zoom
        blob_data.Axis_major_length /= zoom
        blob_data.Axis_minor_length /= zoom
        blob_data.Area /= zoom
        list_zoom_dfs.append(blob_data)
    blob_data = pd.concat(list_zoom_dfs, ignore_index=True)
    # try to remove blobs that appear to be duplicates
    blob_data = filter_close_contacts(blob_data, pixel_dist=2)
    print("zoom-fused blob_data:\n", blob_data)
    print("zoom-fused blob_data.shape:", blob_data.shape)
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
