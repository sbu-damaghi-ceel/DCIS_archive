#!/bin/bash

# This script is used to run the pipeline for the project

#########################################################################
# Step 1: process the raw cell detection of HE & IHC output from Qupath
#process_raw_cellDetection.ipynb

#########################################################################
# step 2: co-register HE with IHCs using VALIS for the same patients
#warpPoints_valis.ipynb

#########################################################################
# step 3: define the niches
# ./cc.sh

#########################################################################
# step 3: extract the features
# python computeMor_text.py
# process_spatialFunc.ipynb
# aggregate_and_split.ipynb

#########################################################################
# step 4: train the classifier model for ROIs of different scales
# python classifier_fromCluster_multiscale.py
