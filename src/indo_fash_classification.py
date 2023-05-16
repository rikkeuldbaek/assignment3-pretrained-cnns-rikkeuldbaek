### Visual Assignment 3
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 14th of April 2023

#--------------------------------------------------------#
################ INDO FASHION CLASSIFIER #################
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# Install packages 
# data wrangeling/path tools/plotting tools 
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt

# data 
import data as dt

# tf tools 
import tensorflow as tf
 
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# call backs
from tensorflow.keras.callbacks import EarlyStopping

#scikit-learn
from sklearn.metrics import classification_report


# Import predefined helper functions (for plotting)
sys.path.append(os.path.join("utils"))
import helper_func as hf

# Scripting
import argparse

###################### PARSER ############################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #add arguments for data.py
    parser.add_argument("--sample_size_train", type=int, default= "16000", help= "Specify sample size of training data.") 
    parser.add_argument("--sample_size_test", type=int, default= "4000", help= "Specify sample size of test data.") 
    parser.add_argument("--sample_size_val", type=int, default= "4000", help= "Specify sample size of validation data.") 
    parser.add_argument("--target_size",nargs='+', type=int, default= (224, 224), help= "Specify target size for image preprocessing.") 
    parser.add_argument("--horizontal_flip", type=bool, default= True, help= "Specify wether the image should be flipped horizontally when agumented.") 
    parser.add_argument("--shear_range", type=float, default= 0.2, help= "Specify the shear angle in counter-clockwise direction in degrees when augmented.") 
    parser.add_argument("--zoom_range", type=float, default= 0.2, help= "Specify range for random zoom when augmented.") 
    parser.add_argument("--rotation_range", type=int, default= 20, help= "Specify degree range for random rotations when augmented.") 
    parser.add_argument("--rescale_1", type=float, default= 1. , help= "Specify ( first digit ) rescaling factor when augmented.") 
    parser.add_argument("--rescale_2", type=float, default= 255. , help= "Specify ( second digit ) rescaling factor when augmented.") 
    parser.add_argument("--batch_size", type=int, default= 32 , help= "Specify size of batch.") 
    parser.add_argument("--n_epochs", type=int, default= 12, help= "Specify number of epochs for model training.") 
    parser.add_argument("--class_mode", type=str, default= "categorical" , help= "Specify class type of target values.") 
    parser.add_argument("--pooling", type=str, default= "avg" , help= "Specify pooling mode for feature extraction.") 
    parser.add_argument("--input_shape",nargs='+', type=int, default= (224, 224, 3) , help= "Specify shape of tuple for feature extraction.") 
    parser.add_argument("--monitor", type=str, default= 'val_loss' , help= "Specify quantity to be monitored.") 
    parser.add_argument("--patience", type=int, default= 5, help= "Specify number of epochs with no improvement after which training will be stopped.") 
    parser.add_argument("--restore_best_weights", type=bool, default= True, help= "Specify whether to restore model weights from the epoch with the best value of the monitored quantity.") 
    parser.add_argument("--nodes_layer_1", type=int, default= 256, help= "Specify number of nodes in first hidden layer.") 
    parser.add_argument("--nodes_layer_2", type=int, default= 128, help= "Specify number of nodes in second hidden layer.") 
    parser.add_argument("--activation_hidden_layer", type=str, default= "relu", help= "Specify activation function to use in hidden layers.") 
    parser.add_argument("--activation_output_layer", type=str, default= "softmax", help= "Specify activation function to use in output layer.") 
    parser.add_argument("--initial_learning_rate", type=float, default= 0.01, help= "Specify the initial learning rate.") 
    parser.add_argument("--decay_steps", type=int, default= 10000, help= "Specify number of decay steps.") 
    parser.add_argument("--decay_rate", type=float, default= 0.9, help= "Specify the decay rate.") 

    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments




######################### IMPORTING DATA ############################
args = input_parse()
test_df, train_df, val_df = dt.load_metadata(args.sample_size_train, args.sample_size_test, args.sample_size_val)


######### DEFINGE DATA GENERATOR ###########

# Specify Image Data Generator
def img_data_generator(horizontal_flip, shear_range, zoom_range, rotation_range, rescale_1, rescale_2):
    
    datagen=ImageDataGenerator(horizontal_flip= horizontal_flip,
                                shear_range= shear_range, # Shear angle in counter-clockwise direction in degrees
                                zoom_range=zoom_range, #Range for random zoom
                                rotation_range=rotation_range, #Degree range for random rotations.
                                rescale=rescale_1/rescale_2) # rescaling factor 
    return(datagen)


# Apply Image Data Generator on data
def generators(datagen, train_df, val_df, test_df, batch_size, target_size, class_mode):

    # Train generator
    train_generator=datagen.flow_from_dataframe(
        dataframe=train_df,
        directory= os.path.join("..","..", ".."), #insert this os.path.join("..", "data")
        x_col="image_path",
        y_col="class_label",
        batch_size=batch_size,
        seed=666,
        shuffle=True,
        class_mode= class_mode,
        target_size=target_size)

    # Validation generator
    val_generator=datagen.flow_from_dataframe(
        dataframe=val_df,
        directory= os.path.join("..","..", ".."), #insert this os.path.join("..", "data")
        x_col="image_path",
        y_col="class_label",
        batch_size=batch_size,
        seed=666,
        shuffle=True,
        class_mode= class_mode,
        target_size=target_size)
    
     # Validation generator
    test_generator=datagen.flow_from_dataframe(
        dataframe=test_df,
        directory= os.path.join("..","..", ".."), #insert this os.path.join("..", "data")
        x_col="image_path",
        y_col="class_label",
        batch_size=batch_size,
        seed=666,
        shuffle=False,
        class_mode= class_mode,
        target_size=target_size)

    return train_generator, val_generator, test_generator



################## DEFINE MODEL ####################

############# loading the model ###############
def loading_model(pooling, input_shape, monitor, patience, restore_best_weights):
    # load the pretrained VGG16 model without classifier layers
    model = VGG16(include_top=False, 
                pooling=pooling, #CHANGE ("max" ?)
                input_shape= input_shape)

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False #resetting 
    # we only wanna update the classification layer in the end,
    # so now we "freeze" all weigths in the feature extraction part and make them "untrainable"

    # Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
    early_stopping = EarlyStopping(monitor = monitor, # watch the val loss metric
                                patience = patience,
                                restore_best_weights = restore_best_weights) # if val loss decreases for 3 epochs in a row, stop training

    return model, early_stopping



########## adding classification layers #########
def add_classification_layers(model, nodes_layer_1, nodes_layer_2, activation_hidden_layer, activation_output_layer, initial_learning_rate, decay_steps, decay_rate):

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1) #normalize weigths
    # 1st layer
    class1 = Dense(nodes_layer_1, 
                activation=activation_hidden_layer)(bn)
    # 2nd layer               
    class2 = Dense(nodes_layer_2, 
                activation=activation_hidden_layer)(class1)
    # output layer    
    output = Dense(15, # n labels
                activation=activation_output_layer)(class2)

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate= initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model



############## FIT & TRAIN #################
def indo_classifier(train_generator, val_generator, batch_size, n_epochs, model, early_stopping): 

    #model
    indo_fash_classifier = model.fit(train_generator,
    steps_per_epoch= train_generator.samples // batch_size,
    epochs = n_epochs,
    validation_data=train_generator,
    validation_steps= val_generator.samples // batch_size,
    batch_size = batch_size, 
    verbose = 1,
    callbacks=[early_stopping])

    return indo_fash_classifier, model


################# EVALUTAION #################
#plotting the model (saves figure in the folder "out")
def evaluation_plot(indo_fash_classifier, n_epochs):
    hf.plot_history(indo_fash_classifier, n_epochs)
    return()


################## PREDICTIONS #################
def indo_fash_predict(model, test_generator, batch_size): 
    predictions = model.predict(test_generator, # X_test
                                batch_size=batch_size)
    return(predictions, test_generator)

############## CLASSIFICATION REPORT ############
def report(test_generator, predictions):  

    # Make classification report
    report=(classification_report(test_generator.classes, # y_test 
                                                predictions.argmax(axis=1),
                                                target_names=test_generator.class_indices.keys())) #labels

    # Define outpath for classification report
    outpath_report = os.path.join(os.getcwd(), "out", "classification_report.txt")
    
    # Save the  classification report
    file = open(outpath_report, "w")
    file.write(report)
    file.close()

    print( "Saving the indo fashion classification report in the folder ´out´")

    return()


#################### MAIN FUNCTION #######################
def main():
    args = input_parse()
    #model arguments
    datagen = img_data_generator(args.horizontal_flip, args.shear_range, args.zoom_range, args.rotation_range, args.rescale_1, args.rescale_2)
    train_generator, val_generator, test_generator = generators(datagen, train_df, val_df, test_df, args.batch_size, tuple(args.target_size), args.class_mode)
    model, early_stopping = loading_model(args.pooling, tuple(args.input_shape), args.monitor, args.patience, args.restore_best_weights)
    model = add_classification_layers(model, args.nodes_layer_1, args.nodes_layer_2, args.activation_hidden_layer, args.activation_output_layer, args.initial_learning_rate, args.decay_steps, args.decay_rate)
    indo_fash_classifier, model = indo_classifier(train_generator, val_generator, args.batch_size, args.n_epochs, model, early_stopping)
    evaluation_plot(indo_fash_classifier, args.n_epochs)
    predictions, test_generator =indo_fash_predict(model, test_generator, args.batch_size)
    report(test_generator, predictions)


if __name__ == '__main__':
    main()
