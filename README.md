# artificial_intelligence

# Phase 1: Data Preparation

Ticket 1: Set Up Project Environment
- Install necessary libraries (e.g., scikit-learn, tensorflow, keras, numpy, matplotlib).
- Set up a Python script or Jupyter Notebook to organize the project.

Ticket 2: Load and Explore the MNIST Dataset
- Load the MNIST dataset using tensorflow.keras.datasets.mnist.
- Explore data structure, inspect a few images, and verify label distribution.

Ticket 3: Preprocess the Data
- Normalize pixel values from 0-255 to 0-1 for both training and testing data.
- Flatten each 28x28 image into a 784-dimensional vector for KNN.

# Phase 2: K-Nearest Neighbors (KNN) Implementation

Ticket 4: Implement K-Nearest Neighbors Classifier
- Use scikit-learn’s KNN model (KNeighborsClassifier).
- Set n_neighbors to a reasonable default, like 3 or 5.

Ticket 5: Train and Test the KNN Model
- Fit the KNN model to the training data.
- Predict digit labels for the test data and evaluate initial accuracy.

Ticket 6: Evaluate KNN Performance
- Calculate accuracy, precision, and recall.
- Use a confusion matrix to visualize misclassifications.
- Experiment with different values of n_neighbors to see if accuracy improves.
