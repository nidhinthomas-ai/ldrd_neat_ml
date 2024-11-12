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

from neat_ml import lib

import glob

import numpy as np
import scipy
import scipy.interpolate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions


def main():
    # Of course the image data is too large to
    # commit to the repo, so it is downloaded from Google Drive
    # by the user independently, before running this code
    data_root_path = "/Users/treddy/LANL/LDRD_DR_NEAT_data/Images"

    # let's find all the % PEO / % DEX .tiff filepaths and do
    # a few sanity checks
    img_filepaths = glob.glob(f"{data_root_path}/**/*.tiff",
                              recursive=True)
    assert len(img_filepaths) == 46

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
    df = lib.opencv_blob_detection(df=df, debug=True)
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_OpenCV_num_blobs",
                                 title_addition="(labels from OpenCV num blobs)",
                                 y_pred=df["num_blobs_opencv"],
                                 norm="symlog",
                                 cbar_label="symlog scaled blob count",
                                 )
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_OpenCV_radii",
                                 title_addition="(labels from OpenCV median radii)",
                                 y_pred=df["median_radii_opencv"],
                                 cbar_label="median OpenCV radii",
                                 )
    points = df[["WT% DEX", "WT% PEO"]].to_numpy()
    values = df["median_radii_opencv"].to_numpy()
    grid_x, grid_y = np.mgrid[0:15:100j, 0:15:100j]
    for method in ["linear", "nearest", "cubic"]:
        interp_vals = scipy.interpolate.griddata(points,
                                                 values,
                                                 (grid_x, grid_y),
                                                 method=f"{method}")
        fig_interp, ax_interp = plt.subplots(1, 1)
        ax_interp.imshow(interp_vals.T, origin='lower', extent=(0, 15, 0, 15))
        ax_interp.set_title(f"PEO/DEX binodal estimation via: {method} interpolation")
        ax_interp.set_ylabel("PEO (wt %)")
        ax_interp.set_xlabel("Dextran (wt %)")
        fig_interp.savefig(f"interp_{method}.png", dpi=300)
    # TODO: perhaps encapsulate the automatic binodal via SVC
    # into an abstracted function, maybe with a test?
    fig_hyper, ax_hyper = plt.subplots(1, 1)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel="rbf"))
    # TODO: less arbitrary threshold for classification...
    threshold = 1
    y = (values > threshold).astype(int)
    clf.fit(points, y)
    plot_decision_regions(X=points,
                          y=y,
                          clf=clf,
                          legend=0,
                          ax=ax_hyper)
    ax_hyper.set_ylabel("PEO (wt %)")
    ax_hyper.set_xlabel("Dextran (wt %)")
    ax_hyper.set_aspect("equal")
    ax_hyper.set_title(f"Automatic Binodal Prototype (OpenCV median radii; {threshold=})\n (method: SVC rbf kernel)")
    fig_hyper.savefig("hyper_opencv.png", dpi=300)




if __name__ == "__main__":
    main()
