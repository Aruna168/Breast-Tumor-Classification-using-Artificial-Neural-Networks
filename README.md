# Breast-Tumor-Classification-using-Artificial-Neural-Networks
Project Overview

This project implements an Artificial Neural Network (ANN) to classify breast tumors as malignant or benign using the Wisconsin Breast Cancer dataset.
It demonstrates a full machine learning workflow, including data preprocessing, model building, training, evaluation, and visualization.

Skills & Tools

Languages: Python

Libraries/Frameworks: TensorFlow / Keras, NumPy, Pandas, Scikit-learn, Matplotlib

Concepts: Neural Networks, Supervised Learning, Data Normalization, Model Evaluation, Visualization

Methodology

Load Dataset: Utilized the built-in Wisconsin Breast Cancer dataset from sklearn.

Preprocessing: Standardized features using StandardScaler; split data into training and test sets.

Build ANN: Feedforward network with input, hidden, and output layers, including Dropout to prevent overfitting.

Train Model: Optimized with Adam, using binary cross-entropy loss for 50 epochs.

Evaluate Model: Generated accuracy, confusion matrix, and classification report on test data.

Visualize Results: Plotted training/validation accuracy and loss curves.

Results

Test Accuracy: ~90–92%

Confusion Matrix & Classification Report: Included in notebook.

Plots:


(Replace with actual plot screenshots from your notebook)

How to Run
# Clone repository
git clone <repo-link>

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook CancerDetection.ipynb

Future Work

Test other models: SVM, Random Forest, Gradient Boosting

Hyperparameter tuning for improved ANN performance

Deploy as web or mobile application for demonstration
