# LDRD_NEAT_ML

## Running the Main ML workflow

Note that the first incantation of the main ML
workflow may take several minutes, but when iterating
or re-running the workflow there are cached operations
that should speed things up (i.e., `pickle` and `joblib`
caching).

Sample incantation: `python main.py --random_seed 42`  

## Running OpenCV Exploration  

`OpenCV_exploration.py` script explores different OpenCV 
parameter settings to detect `hard to find` bubbles 
in the microscopy images. You can change the parameters 
inside the script.

To run the script: 
`python OpenCV_exploration.py `
`--input_dir ./neat_ml/data/OpenCV_exp `
`--base_output_dir ./OpenCV_Exploration_1/ --debug`