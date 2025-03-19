#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path
from typing import Dict, Sequence, List, Union, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
from joblib import Memory
from tqdm import tqdm
import argparse

memory = Memory(location='.', verbose=0)


def build_df_from_exp_img_paths(list_img_filepaths: Sequence[str]) -> pd.DataFrame:
    """
    Build a DataFrame from a list of experimental image file paths.

    Parameters
    ----------
        list_img_filepaths (Sequence[str]):
            A sequence of file paths pointing to image files.

    Returns
    -------
        pd.DataFrame:
            A pandas DataFrame with columns:
                - 'image_filepath': The full path to the image file.
                - 'offset': A float offset value parsed from the filename.
                - 'position': The position string parsed from the filename.
                - 'label': The label string parsed from the filename.
                - 'Contrast': The contrast string parsed from the filename.
    """
    data_dict: Dict[str, List[Union[str, float]]]= {
        "image_filepath": [],
        "offset": [],
        "position": [],
        "label": [],
        "Contrast": []
    }

    for img_path in list_img_filepaths:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        parts = name.split('_')
        offset_str = parts[0].split(' ')[1]
        offset = float(offset_str)
        position = parts[1]
        label = parts[2]
        contrast = parts[4]

        data_dict["image_filepath"].append(img_path)
        data_dict["offset"].append(offset)
        data_dict["position"].append(position)
        data_dict["label"].append(label)
        data_dict["Contrast"].append(contrast)

    df = pd.DataFrame.from_dict(data_dict)
    return df


@memory.cache
def single_run_blob_detection(
    df: pd.DataFrame,
    output_dir: str,
    debug: bool,
    detector_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Run OpenCV SimpleBlobDetector once using the specified parameters.

    This function detects blobs in each image defined in the given DataFrame.
    It optionally saves diagnostic images with detected keypoints overlaid if
    `debug` is True.

    Parameters
    ----------
    df (pd.DataFrame):
        Input DataFrame. Must contain a column 'image_filepath'.
    output_dir (str):
        Directory where output images will be saved if `debug` is True.
    debug (bool):
        Whether to save diagnostic images.
    detector_params (Dict[str, Any]):
        Dictionary of parameter key/value pairs to configure the
        SimpleBlobDetector.

    Returns
    -------
    pd.DataFrame:
        A copy of the input DataFrame with two new columns:
        - 'num_blobs_opencv': The number of detected blobs in each image.
        - 'median_radii_opencv': The median blob radius (pixels) in each
            image, computed from the keypoint size field. If no blobs are
            detected, the value is 0 for the respective row.
    """
    df_new = df.copy()
    median_droplet_radii = np.empty(shape=(df_new.shape[0]), dtype=np.float64)
    num_blobs = np.empty(shape=(df_new.shape[0]), dtype=np.int64)

    params = cv2.SimpleBlobDetector_Params() # type: ignore[attr-defined]

    if "minThreshold" in detector_params:
        params.minThreshold = detector_params["minThreshold"]
    if "maxThreshold" in detector_params:
        params.maxThreshold = detector_params["maxThreshold"]
    if "thresholdStep" in detector_params:
        params.thresholdStep = detector_params["thresholdStep"]
    if "filterByColor" in detector_params:
        params.filterByColor = detector_params["filterByColor"]
    if "blobColor" in detector_params:
        params.blobColor = detector_params["blobColor"]
    if "filterByArea" in detector_params:
        params.filterByArea = detector_params["filterByArea"]
    if "minArea" in detector_params:
        params.minArea = detector_params["minArea"]
    if "maxArea" in detector_params:
        params.maxArea = detector_params["maxArea"]
    if "filterByCircularity" in detector_params:
        params.filterByCircularity = detector_params["filterByCircularity"]
    if "minCircularity" in detector_params:
        params.minCircularity = detector_params["minCircularity"]
    if "maxCircularity" in detector_params:
        params.maxCircularity = detector_params["maxCircularity"]
    if "filterByConvexity" in detector_params:
        params.filterByConvexity = detector_params["filterByConvexity"]
    if "minConvexity" in detector_params:
        params.minConvexity = detector_params["minConvexity"]
    if "maxConvexity" in detector_params:
        params.maxConvexity = detector_params["maxConvexity"]
    if "filterByInertia" in detector_params:
        params.filterByInertia = detector_params["filterByInertia"]
    if "minInertiaRatio" in detector_params:
        params.minInertiaRatio = detector_params["minInertiaRatio"]
    if "maxInertiaRatio" in detector_params:
        params.maxInertiaRatio = detector_params["maxInertiaRatio"]

    detector = cv2.SimpleBlobDetector_create(params)  # type: ignore[attr-defined]

    for index, row in tqdm(df_new.iterrows(), total=df_new.shape[0],
                           desc="SimpleBlobDetector run"):
        blob_radii_img: List[float] = []
        img_filepath = row.image_filepath
        image_basename = Path(img_filepath).stem
        os.makedirs(output_dir, exist_ok=True)

        image = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            median_droplet_radii[index] = np.nan
            num_blobs[index] = 0
            continue

        keypoints = detector.detect(image)  # type: ignore[attr-defined]

        for keypoint in keypoints:
            blob_radii_img.append(keypoint.size)
        blob_radii_img = np.sqrt(np.asarray(blob_radii_img) / np.pi)

        if len(blob_radii_img) == 0:
            blob_radii_img = [0]
        median_droplet_radii[index] = np.median(blob_radii_img)
        num_blobs[index] = len(keypoints)

        if debug:
            fig, axs = plt.subplots(1, 2, figsize=(12, 8))
            image_orig = skimage.color.gray2rgb(image)
            axs[0].imshow(image_orig)
            axs[0].set_title("Original")
            blob_image = cv2.drawKeypoints(
                image.copy(),
                keypoints,
                None,
                (255.0, 0.0, 0.0),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # type: ignore[attr-defined]
            ) # type: ignore[call-overload]
            axs[1].imshow(blob_image)  # type: ignore[arg-type]
            axs[1].set_title(
                f"OpenCV SimpleBlobDetector\nFound {len(keypoints)} blobs"
            )
            output_image_name = f"{image_basename}_detected.png"
            fig.savefig(os.path.join(output_dir, output_image_name), dpi=300)
            plt.close(fig)

    df_new["num_blobs_opencv"] = num_blobs
    df_new["median_radii_opencv"] = median_droplet_radii
    return df_new


def run_parameter_sweep(
    df: pd.DataFrame,
    base_output_dir: str,
    debug: bool = True
) -> None:
    """
    Sweep through multiple sets of SimpleBlobDetector parameters, run the blob
    detection, and save the resulting DataFrame for each set.

    Parameters
    ----------
    df (pd.DataFrame):
        Input DataFrame containing the images to be processed.
    base_output_dir (str):
        The base directory where results (including images, CSV files, etc.)
        will be saved.
    debug (bool, optional):
        Whether to save diagnostic images. Defaults to True.

    Returns
    -------
    None.
    """
    parameter_combinations = [
        {
            "minThreshold": 10,
            "maxThreshold": 200,
            "thresholdStep": 10,
            "filterByColor": False,
            "filterByArea": True,
            "minArea": 50,
            "maxArea": 5000,
            "filterByCircularity": True,
            "minCircularity": 0.6,
            "filterByConvexity": True,
            "minConvexity": 0.8,
            "filterByInertia": True,
            "minInertiaRatio": 0.5
        },
        {
            "minThreshold": 5,
            "maxThreshold": 255,
            "thresholdStep": 20,
            "filterByColor": True,
            "blobColor": 255,
            "filterByArea": True,
            "minArea": 100,
            "maxArea": 10000,
            "filterByCircularity": True,
            "minCircularity": 0.5,
            "filterByConvexity": False,
            "filterByInertia": False
        },
        {
            "minThreshold": 10,
            "maxThreshold": 200,
            "thresholdStep": 10,
            "filterByColor": False,
            "filterByArea": True,
            "minArea": 10,
            "maxArea": 50000,
            "filterByCircularity": True,
            "minCircularity": 0.5,
            "filterByConvexity": True,
            "minConvexity": 0.8,
            "filterByInertia": True,
            "minInertiaRatio": 0.5
        }
    ]

    for i, param_set in enumerate(parameter_combinations, start=1):
        param_dir_name_parts: List[str] = []
        for key, val in param_set.items():
            param_dir_name_parts.append(f"{key}-{val}")
        param_dir_name = "_".join(param_dir_name_parts)[:150]

        output_dir = os.path.join(base_output_dir, f"run_{i}_{param_dir_name}")

        df_result = single_run_blob_detection(df, output_dir, debug, param_set)

        results_csv_path = os.path.join(output_dir, "detection_results.csv")
        df_result.to_csv(results_csv_path, index=False)

        print(f"Finished parameter set {i}/{len(parameter_combinations)}")
        print(f"Parameters: {param_set}")
        print(f"Results saved in: {output_dir}")
        print("-" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run OpenCV blob detection parameter sweep.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./OpenCV_Exploration/",
        help="Path to the directory containing input images."
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="output_blob_sweep",
        help="Base path for saving output data."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to save diagnostic images with detected blobs."
    )

    args = parser.parse_args()

    img_filepaths = glob.glob(
        f"{args.input_dir}/**/*.tiff",
        recursive=True
    )

    df_images = build_df_from_exp_img_paths(img_filepaths)

    run_parameter_sweep(df_images, args.base_output_dir, debug=args.debug)
