#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:37:03 2024

@author: nidhin
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict, Any, Optional, TypeVar
from pathlib import Path
from tqdm import tqdm
import joblib

from .SAM import SAMModel

memory = joblib.Memory("joblib_cache", verbose=0)


def load_image(image_path: str) -> np.ndarray:
    """
    Loads and converts an image from BGR to RGB.

    Parameters
    ----------
    image_path : str
            The path to the image file.

    Returns
    -------
    np.ndarray
            The loaded and converted image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_masks(masks: List[Dict[str, Any]], output_path: str) -> None:
    """
    Saves the generated masks to the specified output path.

    Parameters
    ----------
    masks : List[Dict[str, Any]]
            The generated masks.
    output_path : str
            The path to save the masks.
    """
    for i, mask in enumerate(masks):
        mask_image = (mask["segmentation"] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"mask_{i}.png"), mask_image)


def show_anns(anns: List[Dict[str, Any]]) -> None:
    """
    Shows the mask annotations to an image so that it can be overlayed.

    Parameters
    ----------
    masks : List[Dict[str, Any]]
            The generated masks.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def process_image(
    image_path: str,
    output_dir: str,
    sam_model: SAMModel,
    mask_settings: Dict[str, Any],
    debug: bool = False,
) -> pd.DataFrame:
    """
    Processes a single image, generates a mask, and saves it to the output directory.

    Parameters
    ----------
    image_path : str
            The path to the input image.
    output_dir : str
            The directory to save the masks.
    sam_model : SAMModel
            The initialized SAM model.
    mask_settings : Dict[str, Any]
            Settings for mask generation.
    debug : bool
            If debug = True, then it will save the mask image into output folder.

    Returns
    -------
    PandasDataFrame
            A DataFrame containing properties of the masks (e.g., bubbles).
    """
    image_basename = Path(image_path).stem
    image = load_image(image_path)
    masks = sam_model.generate_masks(output_dir, image, mask_settings)
    masks_summary_DF = sam_model.mask_summary(masks)

    if debug:
        # Determine output path
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks[1:])
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{image_basename}_with_mask.png"))
        plt.close()

    torch.cuda.empty_cache()
    return masks_summary_DF


def bubbleSAM(df: pd.DataFrame, output_dir: str, debug: bool = False) -> pd.DataFrame:
    """
    Main function to perform image segmentation using SAM model.

    Parameters
    ----------
    df : PandasDataFrame
            Input dataframe containing the filepaths.

    output_dir : str
            The directory to save the masks.
    debug : bool
            If debug = True, then it will save the mask image into output folder.

    Returns
    -------
    PandasDataFrame
            Output dataframe containing the original information and additional bubble statistics.
    """
    df_new = df.copy()
    median_droplet_radii = np.empty(shape=(df_new.shape[0]), dtype=np.float64)
    num_blobs = np.empty(shape=(df_new.shape[0]), dtype=np.int64)
    sam_model = SAMModel(
        model_config="sam2_hiera_l.yaml",
        checkpoint_path="neat_ml/sam2/checkpoints/sam2_hiera_large.pt",
        device="cuda",
    )
    mask_settings = {
        "points_per_side": 16,
        "points_per_batch": 128,
        "pred_iou_thresh": 0.90,
        "stability_score_thresh": 0.92,
        "stability_score_offset": 0.7,
        "crop_n_layers": 3,
        "box_nms_thresh": 0.1,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 25,
        "use_m2m": True,
    }
    for index, row in tqdm(df_new.iterrows(), total=df_new.shape[0], desc="bubble_sam"):
        img_filepath = row.image_filepath
        masks_summary_DF = process_image(
            img_filepath, output_dir, sam_model, mask_settings, debug=debug
        )
        num_blobs_img = masks_summary_DF.shape[0]
        median_droplet_radii[index] = np.sqrt(
            np.median(masks_summary_DF["area"]) / np.pi
        )
        num_blobs[index] = num_blobs_img
    df_new["median_radii_SAM"] = median_droplet_radii
    df_new["num_blobs_SAM"] = num_blobs
    df_new.fillna(0, inplace=True)
    print(df_new)
    return df_new


if __name__ == "__main__":

    df_path = "example.csv"
    output_dir = "./output"
    debug = False
    df = pd.read_csv(df_path)
    df_bubble = bubbleSAM(df=df, output_dir=output_dir, debug=debug)
