#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <classifier> <image width> <image width>"
    echo " <classifier>: mlp or svm"
    echo " <width>: a positive integer number"
    echo " <height>: a positive integer number"
    exit 1
fi

# Classifier choice
classifier=$1
width=$2
height=$3

# Check if the classifier choice is valid
if [ "$classifier" != "mlp" ] && [ "$classifier" != "svm" ]; then
    echo "Invalid classifier choice. Please choose 'mlp' or 'svm'."
    exit 1
fi

# Run the Python "Create Dataset" program with the chosen width and height values
python3 create_dataset.py "$width" "$height"
# Run the Python "Classifier" program with the chosen classifier
python3 classifier.py "$classifier"
# Show the correct classified images
display correct_predictions.png