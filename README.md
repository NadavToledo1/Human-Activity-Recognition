# Human Activity Recognition (HAR) Project
This repository contains the complete implementation of a Human Activity Recognition (HAR) system. The project focuses on classifying human activities using data collected from multiple sensors (smartwatch and Vicon sensors). The workflow covers exploratory data analysis (EDA), feature extraction, classical machine learning benchmarking, and advanced deep learning model training.

## Repository Structure
1. EDA.ipynb

Explores the dataset's structure and distribution.

Visualizes sensor data to identify patterns and trends.

Performs statistical analysis and preprocessing steps, including handling missing values, scaling, and feature engineering.

Extracts descriptive features such as mean, standard deviation, RMS, skewness, kurtosis, and correlation coefficients.

2. Training_and_Benchmarking.ipynb

Implements a naïve baseline solution (e.g., class distribution or last-known value prediction).

Extracts relevant features for classical machine learning models, including Random Forests and MLP Classifiers.

Benchmarks classical models to establish performance baselines for deep learning.

3. Models.ipynb

Constructs and trains two neural network architectures:

1D-CNN: Utilizes convolutional layers for feature extraction from time-series data.

LSTM/GRU: Leverages sequential modeling capabilities to capture temporal dependencies.

Visualizes training/validation metrics (e.g., accuracy, loss curves).

Analyzes results, including misclassified examples, uncertain predictions, and classification confidence.

## Dataset

The project uses a dataset containing sensor data from two sources:

1. Smartwatch: Records acceleration, magnetometer, and gyroscope measurements along three axes (x, y, z).
   
2. Vicon: Tracks 3D position measurements along the x, y, and z axes.

### Data Highlights:
Sequence Lengths: Vary between 3000, 3500, and 4000.

Features: Includes raw sensor data and engineered features such as FFT, ACF, skewness, and kurtosis.

Labels: Activities classified into predefined categories (e.g., walking, running).

## Key Features

**Exploratory Data Analysis:**

- Statistical summaries and visualizations to understand data distribution.
- Feature extraction and engineering using domain knowledge and statistical techniques.

**Baseline Benchmarking:**

- Naïve solutions to establish a starting point.
  
- Classical machine learning models for robust benchmarking.
  
**Deep Learning Models:**

- 1D-CNN for localized feature extraction.
  
- LSTM/GRU models for sequential and temporal pattern recognition.
  
- Extensive evaluation and comparison of models using metrics like accuracy, confusion matrix, and log-loss.

  ## Installation
### Clone the repository
    git clone https://github.com/NadavToledo1/Human-Activity-Recognition.git  
    cd HAR-Project  

### Install Required Libraries
Use the requirements.txt file to install dependencies:

    pip install -r requirements.txt 

### Acknowledgments
- Sensor data for the project was provided by the Human Activity Recognition dataset.
- Deep learning libraries such as PyTorch and TensorFlow were pivotal in building the models.
- The project benefitted from tools like sklearn, statsmodels, and seaborn for analysis and visualization.

### Future Work
- Integrating advanced feature extraction techniques (e.g., tslearn, sktime).
- Optimizing model architectures with more extensive hyperparameter tuning.
- Exploring ensemble methods to combine classical and deep learning models.
  
Feel free to contribute to this repository by submitting pull requests or opening issues.
