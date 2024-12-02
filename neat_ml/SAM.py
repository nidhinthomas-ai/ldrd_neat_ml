#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:37:49 2024

@author: nidhin
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAMModel:
    def __init__(self, model_config: str, checkpoint_path: str, device: str) -> None:
        """
        Initializes the SAM model with the given checkpoint and device.

        Parameters
        ----------
        model_config : str
                The config of the model.
        checkpoint_path : str
                The path to the model checkpoint.
        device : str
                The device to run the model on.
        """
        self.model_config = model_config
        self.checkpoint = checkpoint_path
        self.device = device

    def setup_cuda(self) -> None:
        """
        Set up CUDA environment with bfloat16 and tfloat32 for Ampere GPUs.
        """
        # Use bfloat16 for the entire script
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # Turn on tfloat32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def generate_masks(
        self,
        output_dir: str,
        image: np.ndarray,
        mask_settings: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image using the SAM model.

        Parameters
        ----------
        output_dir : str
                The output directory where masks will be saved
        image : np.ndarray
                The image to generate masks for.
        mask_settings : Dict[str, Any], optional
                Additional settings for the mask generator.

        Returns
        -------
        List[Dict[str, Any]]
                The generated masks.
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sam2 = build_sam2(
            self.model_config,
            self.checkpoint,
            device=self.device,
            apply_postprocessing=False,
        )
        mask_generator = SAM2AutomaticMaskGenerator(model=sam2, **mask_settings)
        masks: List[Dict[str, Any]] = mask_generator.generate(image)
        sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        return sorted_masks

    def mask_summary(self, masks: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        summarizes and returns properties of the masks such as count, area, and bounding box.

        Parameters
        ----------
        masks : List[Dict[str, Any]]
                The generated masks.

        Returns
        -------
        PandasDataFrame
                A dataframe containing the properties of the masks.
        """
        keys_to_remove = [
            "crop_box",
            "segmentation",
            "bbox",
            "predicted_iou",
            "stability_score",
            "point_coords",
        ]
        mask_dict = [
            {k: v for k, v in d.items() if k not in keys_to_remove} for d in masks
        ]
        masks_DF = pd.DataFrame(mask_dict)

        return masks_DF
