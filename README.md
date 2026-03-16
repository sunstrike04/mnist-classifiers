# MNIST Image Classification from Scratch

This project implements fundamental machine learning classifiers from scratch using NumPy to recognize handwritten digits from the MNIST dataset. It was developed as part of the Deep Learning for Computer Vision course (CS 444) at the University of Illinois Urbana-Champaign (UIUC).

## Features

*   **K-Nearest Neighbors (k-NN):** Implemented with fully vectorized L2 distance computation for high performance.
*   **Linear Classifier (Softmax):** Multiclass logistic regression trained via batch gradient descent with L2 regularization and a numerically stable softmax implementation.
*   **Feature Engineering:**
    *   **Raw Pixels:** Flattened 28x28 images.
    *   **Mean Pooling:** Dimensionality reduction via mean pixel values over non-overlapping grid cells.
    *   **Histogram of Oriented Gradients (HOG):** Captures edge and shape information by binning gradient orientations within image patches.

## Setup

It is recommended to use a Conda environment to run this project:

```bash
conda create --name dlcv_mp1 python=3.10 -y
conda activate dlcv_mp1
pip install -r requirements.txt
```

## Usage

Use the `demo.py` script to train and evaluate the models. The MNIST dataset will be downloaded automatically the first time you run a command.

**Run Linear Classifier with HOG features:**
```bash
python demo.py --classifier linear --lr 1e-1 --wt 1e-3 --feature hog --pool_size 4 --num_train 1000 --out_dir runs/linear-hog4
```

**Run k-NN Classifier with raw pixels:**
```bash
python demo.py --classifier knn --k 5 --num_train 1000 --feature raw --out_dir runs/knn-raw
```

## Results

The Linear Classifier utilizing Histogram of Oriented Gradients (HOG) features achieved the highest validation accuracy (~94%), successfully demonstrating the importance of feature engineering in traditional machine learning pipelines.