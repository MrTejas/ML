# K-Nearest Neighbours

This directory contains the implementation of KNN with various performance measurements, hyperparameter tuning and trends graphs. The directory follows the following structure :

## Files
KO
Following are the files in this directory :-
- `KNN.ipynb` : contains the main implementation of KNN. It contains the KNN class and all the parts of the questions in the assignment.
- `results.txt` contains the performance results for all the possible combinations of hyperparameters provided in the assignment
- `sort.ipynb` : This notebook contains just one function to sort the top 20 results in `results.txt` by some column (mostly accuracy) and store them into `results_final.txt`
- `run.sh` is a bash script that takes input a dataset file (<dataset_name>.npy) and runs the KNN for fixed set of hyperparameters and prints the performance. It uses `run_KNN.py` file to do so.
- `run_KNN.py` file imports the `KNN_class.py` (which is just a python file of the KNN class) and creates an object on that class to run the model 


## Running `run.sh`

- Firstly give permission to run `run.sh` on your system using the command `chmod +x run.sh` on the terminal of the same directory
- Then you can run the shell script using the format `./run.sh <filename>.npy` where filename is the dataset numpy file.

