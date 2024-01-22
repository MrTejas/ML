# Decision Tree

This directory contains wrapper class for sklearn's Decision Tree with various performance measurements, hyperparameter tuning and some explotary data visualization. The directory follows the following structure :

## Files
KO
Following are the files in this directory :-
- `DecisionTree.ipynb` : contains the main implementation of wrapper for DecisionTree. It contains the DecisionTree class and all the parts of the questions in the assignment.
- `results.txt` contains the performance results for all the possible combinations of hyperparameters provided in the assignment
- `sort.ipynb` : This notebook contains just one function to sort the results in `results.txt` by some column (mostly f1(macro)) and store them into `results_final.txt`
- `results_final.txt` contains the top 3 set of hyper-parameters for each of the 2 formulation :- ***Powerset and MultiOutput*** 
- `visualization.ipynb` contains the code for data visualization of the provided dataset (this part has been written with reference from Generative AI)
