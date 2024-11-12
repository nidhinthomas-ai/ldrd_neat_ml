# LDRD_NEAT_ML

## Installation

Download the package from the following GitLab repository:  

```bash
git clone --recurse-submodules git@gitlab.lanl.gov:treddy/ldrd_neat_ml.git  

cd ldrd_neat_ml/sam2

git rev-list -n 1 --before="2024-08-06" main

cd ../
```

To set up the environment:  

```bash
conda env create -n ldrd_neat_ml "python>=3.10"
conda activate ldrd_neat_ml

conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

To set up the environment for Segment Anything-2, refer to `INSTALL.md` in segment-anything-2 
repository and ensure that you have access to GPUs for better performance, when running the script.  

If the following ImportError `ImportError: cannot import name '_C' from 'sam2'` occurs during the run, do the following:  

```bash
cd sam2

python setup.py build_ext --inplace
```  

To run the code in CHICOMA:  
```bash
module load intel-classic/2022.2.1 lanlpe/legacy friendly-testing cuda/11.7  
```  

## Detecting Bubbles using SAM-2

Here, we detect bubbles from microscopic images. We use the `sam2_hiera_large.pt` as the
checkpoint file. The instructions to download the checkpoint file are provided in
`download_ckpts.sh` in `sam2/checkpoints` subdirectory. The parameters used for 
detecting the bubbles are given in `bubbleSAM()` method in `bubble_sam/bubblesam.py`.  
In order to run the script faster, modify the `points_per_side` to 16.  

## Running the Main ML workflow

Note that the first incantation of the main ML
workflow may take several minutes, but when iterating
or re-running the workflow there are cached operations
that should speed things up (i.e., `pickle` and `joblib`
caching).

Sample incantation: `python main.py --random_seed 42`


