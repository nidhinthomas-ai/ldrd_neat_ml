import pytest
import numpy as np
import os
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from neat_ml.bubblesam import (
    load_image,
    save_masks,
    show_anns,
    process_image,
    bubbleSAM,
)


@pytest.mark.parametrize(
    "image_exists, expected_exception",
    [
        (True, None),
        (False, FileNotFoundError),
    ],
)
def test_load_image(image_exists, expected_exception):
    """
    Test the functionality of the load_image function.

    - Verifies that a valid image file is loaded correctly,
      with shape and pixel values matching the original image.
    - Checks if a FileNotFoundError is raised when the image file does not exist.
    """
    test_image_path = "test_image.png"

    if image_exists:
        sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
        sample_image[0, 0] = [0, 0, 255]
        cv2.imwrite(test_image_path, sample_image)

        try:
            loaded_image = load_image(test_image_path)
            assert loaded_image.shape == sample_image.shape
            assert np.allclose(loaded_image[0, 0], [255, 0, 0], atol=1)
        finally:
            os.remove(test_image_path)
    else:
        with pytest.raises(expected_exception):
            load_image("non_existent_image.jpg")


@pytest.mark.parametrize(
    "masks",
    [
        ([{"segmentation": np.zeros((100, 100), dtype=bool)}]),
        ([{"segmentation": np.ones((100, 100), dtype=bool)}]),
        ([{"segmentation": np.eye(100, dtype=bool)}]),
        ([]),
    ],
)
def test_save_masks(tmp_path, masks):
    """
    Test the save_masks function.

    - Ensures that mask images are saved correctly when a list of masks is provided.
    - Checks the output files for the correct pixel data.
    - Verifies no files are created if the mask list is empty.
    """
    output_path = tmp_path / "masks"
    os.makedirs(output_path, exist_ok=True)

    save_masks(masks, str(output_path))

    expected_files = [output_path / f"mask_{i}.png" for i in range(len(masks))]
    for i, file_path in enumerate(expected_files):
        assert file_path.exists()
        mask_image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        assert mask_image is not None
        expected_image = (masks[i]["segmentation"] * 255).astype(np.uint8)
        assert np.array_equal(mask_image, expected_image)

    if not masks:
        assert not any(output_path.iterdir())


@pytest.mark.parametrize(
    "anns",
    [
        (
            [
                {"segmentation": np.random.rand(100, 100) > 0.5, "area": 5000},
                {"segmentation": np.random.rand(100, 100) > 0.7, "area": 3000},
            ]
        ),
        ([]),
        (
            [
                {"segmentation": np.ones((100, 100), dtype=bool), "area": 10000},
            ]
        ),
    ],
)
def test_show_anns(anns):
    """
    Test the show_anns function.

    - Verifies that the function correctly overlays annotations on an image.
    - Ensures that when no annotations are provided, no images are shown.
    """
    plt.figure()
    show_anns(anns)
    ax = plt.gca()
    images = ax.get_images()
    if anns:
        assert len(images) > 0
    else:
        assert len(images) == 0
    plt.close()


@pytest.mark.parametrize(
    "image_path, debug",
    [
        ("test_image.jpg", False),
        ("test_image.jpg", True),
    ],
)
def test_process_image(monkeypatch, image_path, debug, tmp_path):
    """
    Test the process_image function.

    - Verifies that masks are generated and returned in the correct DataFrame format.
    - Checks that debug mode saves additional output files as expected.
    """

    class MockSAMModel:
        def generate_masks(self, output_dir, image, mask_settings):
            return [{"segmentation": np.zeros((100, 100), dtype=bool), "area": 100}]

        def mask_summary(self, masks):
            return pd.DataFrame({"area": [100], "other_property": [42]})

    mock_sam_model = MockSAMModel()

    def mock_load_image(image_path):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    monkeypatch.setattr("neat_ml.bubblesam.load_image", mock_load_image)

    output_dir = tmp_path / "output"
    masks_summary_DF = process_image(
        image_path, str(output_dir), mock_sam_model, {}, debug=debug
    )

    assert isinstance(masks_summary_DF, pd.DataFrame)
    assert "area" in masks_summary_DF.columns
    assert masks_summary_DF.iloc[0]["area"] == 100

    if debug:
        image_basename = Path(image_path).stem
        expected_image_path = output_dir / f"{image_basename}_with_mask.png"
        assert expected_image_path.exists()


@pytest.mark.parametrize(
    "df_data",
    [
        ({"image_filepath": ["image1.jpg", "image2.jpg"]}),
        ({"image_filepath": ["image1.jpg"]}),
        ({"image_filepath": []}),
    ],
)
def test_bubbleSAM(monkeypatch, df_data):
    """
    Test the bubbleSAM function.

    - Ensures that the function processes all rows in the input DataFrame.
    - Verifies that median radii and blob counts are correctly calculated.
    - Handles cases with empty or non-empty input DataFrames.
    """
    df = pd.DataFrame(df_data)

    def mock_process_image(
        image_path, output_dir, sam_model, mask_settings, debug=False
    ):
        num_masks = len(image_path)
        areas = [100 * (i + 1) for i in range(num_masks)]
        return pd.DataFrame({"area": areas})

    monkeypatch.setattr("neat_ml.bubblesam.process_image", mock_process_image)

    class MockSAMModel:
        pass

    monkeypatch.setattr("neat_ml.SAM.SAMModel", MockSAMModel)

    output_df = bubbleSAM(df, output_dir="test_output_dir", debug=False)

    assert "median_radii_SAM" in output_df.columns
    assert "num_blobs_SAM" in output_df.columns
    assert len(output_df) == len(df)
    for index, row in output_df.iterrows():
        areas = (
            [100 * (i + 1) for i in range(len(row["image_filepath"]))]
            if row["image_filepath"]
            else []
        )
        expected_median_radius = np.sqrt(np.median(areas) / np.pi) if areas else 0
        np.testing.assert_almost_equal(row["median_radii_SAM"], expected_median_radius)
        expected_num_blobs = len(areas)
        assert row["num_blobs_SAM"] == expected_num_blobs
