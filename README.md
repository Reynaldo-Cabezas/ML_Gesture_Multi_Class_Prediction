# ML_Gesture_Multi_Class_Prediction
Machine learning-based gesture phase segmentation using Kinect data. Classifies each video frame into one of five gesture phases. Includes baseline model evaluation, hyperparameter tuning, and visualizations. Optimized for Colab and external GPU use (e.g., AWS EC2 G5).
# Gesture Phase Segmentation

## About Us

This project was developed by **Reynaldo Cabezas** and **Fabricio Meneses**, Master’s students in **Business Intelligence and Data Analysis** at **Universidad Americana (UAM)**, Nicaragua. We both contributed to the design, implementation, and evaluation of machine learning models for gesture phase classification. Our goal is to apply data-driven approaches to real-world problems in human-computer interaction.

## Project Overview

This repository contains code and documentation for a machine learning project focused on automatic gesture phase segmentation using a dataset collected via Microsoft Kinect. The dataset includes spatial, velocity, and acceleration features derived from motion data, annotated with five gesture phases: Rest, Preparation, Stroke, Hold, and Retraction.

## Dataset Source

**Creators:**

* Renata Cristina Barros Madeo  
* Priscilla Koch Wagner  
* Sarajane Marques Peres

**Donor:**

* University of Sao Paulo, Brazil

The dataset consists of seven videos recorded by three individuals narrating comic strips. Each video is annotated by specialists to mark the gesture phases. Data includes both raw 3D coordinates and processed velocity and acceleration features.

## Project Goals

The goal of this project is to develop models capable of segmenting gesture phases in real time based on Kinect-captured data. This is crucial for applications in sign language interpretation, assistive technologies, and natural human-computer interaction.

## Preprocessing and Feature Engineering

* Merged raw and processed files for each video  
* Extracted velocity and acceleration features for hands and wrists  
* Normalized joint positions relative to the head and spine  
* Treated outliers using Z-score filtering (Z > 3)  
* Created additional features by summing, subtracting, and multiplying multimodal variables  
* Final feature set includes up to 100 attributes

## Models Evaluated

We trained and evaluated the following models:

* Decision Trees  
* Random Forests  
* Extra Trees  
* Support Vector Machines (SVM)  
* Naive Bayes  
* Perceptron  
* Gradient Boosting  
* XGBoost

## Model Evaluation and Cross-Validation

* Stratified 5-fold cross-validation used for all evaluations  
* F1 score (macro and weighted) used as the primary performance metric  
* Time taken for training and prediction recorded

## Hyperparameter Tuning

Focused on the best-performing models: Random Forests, Extra Trees, Gradient Boosting, and XGBoost

* Used Grid Search, Random Search, and Bayesian Optimization  
* Hyperparameters tuned included number of estimators, minimum samples per split, and leaf size  
* Visualizations include F1 scores and training durations

## Limitations

The main limitation was computational power:

* Google Colab Pro often failed to provide access to GPUs, defaulting to CPU only  
* Long training sessions were interrupted due to inactivity timeouts  
* Cell execution was frequently restarted, leading to lost progress  
* An attempt was made to use an AWS EC2 G5.xlarge instance with an A10G GPU, but delays in approval pushed us to continue on Colab  
* This constrained the extent of hyperparameter search and forced trade-offs between parameter grid size and training time

## Conclusion

This project demonstrates the feasibility of automated gesture phase segmentation using classical and ensemble machine learning models. The task remains challenging due to phase ambiguity and movement overlap, but promising results were obtained. Accurate multiclass prediction in this context has high potential for enabling more natural, real-time multimodal systems.


## Conclusion

This project demonstrates the feasibility of automated gesture phase segmentation using classical and ensemble machine learning models. The task remains challenging due to phase ambiguity and movement overlap, but promising results were obtained. Accurate multiclass prediction in this context has high potential for enabling more natural, real-time multimodal systems.
