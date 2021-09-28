#Deep VASP-E: A Flexible Analysis of Electrostatic Isopotentials for Finding and Explaining Mechanisms that Control Binding Specificity


## GRAD++ CAM


## Project structure
####Models
Models package is responsible in holding different models being used for different projects. Note, models do not do any 
hypertuning of hyperparameters. Keras has tuners available. 

####processing
Package is responsible for preprocessing dataset and postprocessing model results (Grad CAM++).

## Dataset structure

Dataset structure for loading in training and test set is done with the following format:
                `Dataset_directory/protein001/protein001_example001.CNN`
Currently, implementation requires code to specify the evaluation set following a leave one out approach.
    