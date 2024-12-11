# CNN-for-Image-Classification
This repository contains all the code and report for my CNN-based image classification project.

1. Installations
This project was written in Python, using Jupyter Notebook. The relevant Python packages for this project are as follows:

pandas
numpy
tensorflow (integrated with Keras)
matplotlib
sklearn.model_selection (train_test_split module)
sklearn.metrics (classification_report)

Ensure you have these packages installed before running the code. You can install the dependencies using the command:
pip install pandas numpy tensorflow matplotlib scikit-learn

2. Project Motivation
This project is focused on building a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset includes:
Training Data: 4000 images of dogs and 4000 images of cats.
Testing Data: 1000 images of dogs and 1000 images of cats for evaluation.

The main objectives of this project are:
Train a CNN to classify images as either "cat" or "dog."
Test the trained model on unseen data to evaluate its performance.
Provide functionality to classify new images provided in the "single_prediction" folder.
The project leverages TensorFlow's integration with Keras, simplifying the process of designing and training the CNN model.

3. File Descriptions
This project contains the following files and folders:

CNN_Image_Classification.ipynb: The main Jupyter Notebook file containing the code and analysis.
dataset/: Contains the training and testing datasets.
single_prediction/: A folder where new images can be placed for model prediction.
README.md: Documentation for the project.


4. Results
The trained CNN model achieved high accuracy in classifying images of cats and dogs. The main findings of this project are:

Training Dataset: 4000 images of cats and 4000 images of dogs.
Testing Dataset: 1000 images of cats and 1000 images of dogs.
Model Performance:
High accuracy was achieved on both the training and testing datasets.
The model performed well in classifying new images placed in the "single_prediction" folder.
Further details can be found in the accompanying Medium post: Medium Article link.

5. Licensing, Authors, Acknowledgements, etc.
Code and documentation were developed by the project author.
Acknowledgements to TensorFlow and Keras for providing tools for deep learning.
