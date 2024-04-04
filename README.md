# Dragons vs Wolves Classification Project

## Description
This project implements a simple image classification system using machine learning classifiers (SVM and MLP) in Python. It allows users to train and evaluate classifiers on a dataset of images. The project also provides a shell script to simplify the process of running the classifiers.

## Dependencies
- Python 3.x
- scikit-learn
- numpy
- matplotlib
- [OPTIONAL] ImageMagick (for displaying images from the shell script)

## Installation
1. **Python**: If you don't have Python installed, download and install it from [python.org](https://www.python.org/).

2. **scikit-learn, numpy, matplotlib**: Install these dependencies using pip:
   ```bash
   pip install scikit-learn numpy matplotlib
   ```

3. **ImageMagick** (Optional): Install ImageMagick using your package manager. For example, on Debian-based systems:
   ```bash
   sudo apt-get update
   sudo apt-get install imagemagick
   ```

## How to Use
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/lopesvictor1/Dragons-vs-Wolves-Classification.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Dragons-vs-Wolves-Classification
   ```

3. Run the shell script `classify.sh` to train and evaluate classifiers. You can pass `mlp` or `svm` as an argument to specify the classifier:
   ```bash
   bash classify.sh <classifier> <width> <height>
   ```

   Replace `<classifier>` with `svm` if you want to use the Support Vector Machine classifier or `mlp` if you want to use the Multi-Layer Perceptron classifier.
   Replace `width` and `height` with the image parameters you prefer. *Note: Values for `width` and `height` higher than 128 could cause the classifiers to take a long time in the fitting proccess* 

4. Follow the prompts or check the output for the results of the classification process.

## Example
Here's an example of running the shell script to train and evaluate the MLP classifier:
   ```bash
   bash classify.sh mlp 32 32
   ```

