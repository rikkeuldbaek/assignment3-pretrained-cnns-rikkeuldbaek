### Visual Assignment 3
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 14th of April 2023

#--------------------------------------------------------#
################## DATA PREPROCESSING  ###################
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# Import packages
# tf tools 
import tensorflow as tf
import pandas as pd
import os
import numpy as np

# Scripting
import argparse

###################### PARSER ############################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #add arguments for data.py
    parser.add_argument("--sample_size_train", type=int, default= "8000", help= "Specify sample size of training data.") 
    parser.add_argument("--sample_size_test", type=int, default= "2000", help= "Specify sample size of test data.") 
    parser.add_argument("--sample_size_val", type=int, default= "2000", help= "Specify sample size of validation data.") 
    parser.add_argument("--target_size",nargs='+', type=int, default= (224, 224), help= "Specify target size for image preprocessing.") 


    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments




##################### PREPROCESSING META DATA  ######################
# Load the metadata .json files into a pandas dataframe
def load_metadata(sample_size_train, sample_size_test, sample_size_val):
    print("Loading metadata into a pandas dataframe..")
    dfs = {}
    for x in ['test_data.json', 'train_data.json', 'val_data.json']:
        dfs[x] = pd.read_json(r"../../../images/metadata/%s" % x, lines=True) # Quote this out
        #dfs[x] = pd.read_json(r"../data/images/metadata/%s" % x, lines=True) # Unquote this path

    print("Sampling "+ str(sample_size_train)+ " data points randomly from the training data..")
    print("Sampling "+ str(sample_size_test)+ " data points randomly from the test data..")
    print("Sampling "+ str(sample_size_val)+ " data points randomly from the validation data..")

    # construct dataframes
    test_df  = dfs['test_data.json'].sample(n=sample_size_test, random_state=1)
    train_df= dfs['train_data.json'].sample(n=sample_size_train, random_state=1)
    val_df = dfs['val_data.json'].sample(n=sample_size_val, random_state=1)
    
    return test_df, train_df, val_df




#################### MAIN FUNCTION #######################
def main():
    args = input_parse()
    test_df, train_df, val_df = load_metadata(args.sample_size_train, args.sample_size_test, args.sample_size_val)

if __name__ == '__main__':
    main()
