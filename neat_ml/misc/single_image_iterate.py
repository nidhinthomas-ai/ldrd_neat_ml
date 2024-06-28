"""
The purpose of this module is to replicate key
parts of the main image analysis workflow, but
apply them only to a single plate reader image
for faster iteration/improvement of bubble
detection/analysis methods.

Sample incantation:
time python neat_ml/misc/single_image_iterate.py --image-path '/home/treddy/LANL/LDRD_DR_NEAT_data/Images/DEXTRAN (10k) 2~14wt_ (with PEO 10K)/DEXTRAN 12wt_/DEX12wt_,PEO10wt_.tiff'
"""

import argparse
import time

import cv2
import skimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="Path to the plate reader image file to be processed")
    args = parser.parse_args()
    opencv_blob_detection_single_image(image_path=args.image_path,
                                       debug=True)
