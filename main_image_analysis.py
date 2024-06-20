"""
Given an image captured for a % PEO / % DEX
mixture, try to automatically determine
the (average?) diameter of droplets present
in the image, as a proxy for phase separation.

Presumably, we'll use a diameter of `0` to indicate
absence of relevant droplets.

Part of the purpose here is to avoid human bias
in the generation of the response variables for
our downstream ML work.
"""

import argparse
import glob

from neat_ml import lib

import numpy as np


def main(data_root_path: str):
    # let's find all the % PEO / % DEX .tiff filepaths and do
    # a few sanity checks
    img_filepaths = glob.glob(f"{data_root_path}/**/*.tiff",
                              recursive=True)
    num_image_file_paths = len(img_filepaths)
    expected_image_file_paths = 46
    msg = f"Expected {expected_image_file_paths} image file paths, but found {num_image_file_paths}"
    assert num_image_file_paths == expected_image_file_paths, msg

    # check that images all have the same pixel dims
    # (2456 x 2052 at the time of writing)
    lib.check_image_dim_consistency(img_filepaths)

    # Parse out the (WT % DEX, WT % PEO) information
    # from the image filepaths, and start building relevant
    # information into a DataFrame
    df = lib.build_df_from_exp_img_paths(img_filepaths)
    assert df.shape == (46, 3)
    # had a bug where PEO/DEX columns were accidentally
    # the same, so rule that out:
    diff = df["WT% PEO"] - df["WT% DEX"]
    assert not np.allclose(diff, np.zeros(df.shape[0]))

    # Produce a standard plot of the binary phase system
    # points (we don't have phase "labels" yet)
    # TODO: should rename this function to something more generic
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_",
                                 )

    # there are a variety of ways we could try to estimate
    # the droplet sizes (diameters); perhaps it makes sense to try a few
    # and compare them

    # 1) Using the Hough Transform

    df = lib.skimage_hough_transform(df=df, debug=True)
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_hough_",
                                 title_addition="(labels from median Hough radii)",
                                 y_pred=df["median_radii_skimage_hough"],
                                 cbar_label="median Hough radii",
                                 )

    # 2) Using Blob Detection Techniques
    df = lib.blob_detection(df=df, debug=True)
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_DoH_sigma",
                                 title_addition="(labels from median DoH sigma/radii)",
                                 y_pred=df["median_radii_DoH"],
                                 cbar_label="median DoH sigma",
                                 )
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_DoH_num_blobs",
                                 title_addition="(labels from DoH num blobs)",
                                 y_pred=df["num_blobs_DoH"],
                                 norm="symlog",
                                 cbar_label="symlog scaled blob count",
                                 )
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_LoG_num_blobs",
                                 title_addition="(labels from LoG num blobs)",
                                 y_pred=df["num_blobs_LoG"],
                                 norm="symlog",
                                 cbar_label="symlog scaled blob count",
                                 )
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_LoG_radii",
                                 title_addition="(labels from median LoG radii)",
                                 y_pred=df["median_radii_LoG"],
                                 cbar_label="median LoG radii",
                                 )
    # try with OpenCV as well:
    df_opencv = lib.opencv_blob_detection(df=df, debug=True)
    lib.plot_input_data_cesar_MD(df=df_opencv,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_OpenCV_num_blobs",
                                 title_addition="(labels from OpenCV num blobs)",
                                 y_pred=df_opencv["num_blobs_opencv"],
                                 norm="symlog",
                                 cbar_label="symlog scaled blob count",
                                 )
    lib.plot_input_data_cesar_MD(df=df_opencv,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_OpenCV_radii",
                                 title_addition="(labels from OpenCV median radii)",
                                 y_pred=df_opencv["median_radii_opencv"],
                                 cbar_label="median OpenCV radii",
                                 )
    # try with Kim and Park pre-trained deep learning model:
    df_kim_park = lib.kim_park_dl_blob_detection(df=df, debug=True)
    lib.plot_input_data_cesar_MD(df=df_kim_park,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_Kim_Park_num_blobs",
                                 title_addition="(labels from Kim and Park num blobs)",
                                 y_pred=df_kim_park["num_blobs_kim_park"],
                                 norm="symlog",
                                 cbar_label="symlog scaled blob count",
                                 )
    lib.plot_input_data_cesar_MD(df=df_kim_park,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_Kim_Park_area",
                                 title_addition="(labels from Kim and Park median area)",
                                 y_pred=df_kim_park["median_droplet_area_kim_park"],
                                 cbar_label="median area (Kim and Park)",
                                 )
    # plot the standard deviations of the blob counts
    # across different blob detection methods, to try
    # to assess PEO/DEX locations of disagreement
    list_series = [df_kim_park["num_blobs_kim_park"],
                   df_opencv["num_blobs_opencv"],
                   df["num_blobs_LoG"],
                   df["num_blobs_DoH"],
                  ]
    df_std = lib.produce_df_num_blobs_std_dev(list_series=list_series)
    # TODO: handle this in lib code probably:
    df_std["WT% DEX"] = df_kim_park["WT% DEX"]
    df_std["WT% PEO"] = df_kim_park["WT% PEO"]
    lib.plot_input_data_cesar_MD(df=df_std,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_std_dev_of_counts",
                                 title_addition="(labels from standard deviations of blob counts across methods)",
                                 y_pred=df_std["std_dev_num_blobs"],
                                 norm="symlog",
                                 cbar_label="symlog standard dev of blob counts",
                                 )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Of course the image data is too large to
    # commit to the repo, so it is downloaded from Google Drive
    # by the user independently, before running this code
    parser.add_argument("--root-path", type=str, help="Root path of the experimental plate reader image data")
    args = parser.parse_args()
    data_root_path = args.root_path
    main(data_root_path=data_root_path)
