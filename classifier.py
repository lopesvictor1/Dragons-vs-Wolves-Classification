import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt





def svm_classifier(X_train, X_test):
    # Flatten the images for SVM classifier
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Create and train the SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_flat, y_train)

    # Predict labels for test data
    y_pred = svm_classifier.predict(X_test_flat)
    return y_pred


def mlp_classifier(X_train, X_test):
    # Flatten the images for MLP classifier
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Create and train the MLP classifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(128,64, 32), max_iter=1000, verbose = True)
    mlp_classifier.fit(X_train_flat, y_train)

    # Predict labels for test data
    y_pred = mlp_classifier.predict(X_test_flat)
    return y_pred




if __name__ == '__main__':
    sys.argv = sys.argv[1:]
    if len(sys.argv) != 1:
        print(len(sys.argv))
        print("Usage: python classifier.py <'svm' or 'mlp'>")
        sys.exit(1)
    folder = os.getcwd()
    print(sys.argv)

    list_folder = sorted(os.listdir(folder))
    class_folders = []
    for class_label, class_name in enumerate(class_folders):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            class_folders.append(class_name)

    # Load saved images and labels
    images = np.load("images.npy")
    labels = np.load("labels.npy")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


    if sys.argv[0] == 'mlp':
        y_pred = mlp_classifier(X_train, X_test)
    elif sys.argv[0] == 'svm':
        y_pred = svm_classifier(X_train, X_test)

    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_folders, yticklabels=class_folders)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    misclassified_indices = np.where(y_pred != y_test)[0]

    # Show misclassified images
    for index in misclassified_indices:
        print("True Label:", list_folder[y_test[index]], ", Predicted Label:", list_folder[y_pred[index]])
        plt.imshow(X_test[index])
        plt.axis('off')
        plt.title("True Label:" + list_folder[y_test[index]] + ", Predicted Label:" + list_folder[y_pred[index]])
        plt.show()

    # Create a figure to aggregate correct predictions
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(20, 10))

    # Initialize counter for subplot indices
    row_index = 0
    col_index = 0

    # Iterate through test set
    for i, (image, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
        if true_label == pred_label:
            # Plot the image
            axes[row_index, col_index].imshow(image)
            axes[row_index, col_index].axis('off')
            axes[row_index, col_index].set_title(f"True: {list_folder[true_label]}\nPredicted: {list_folder[pred_label]}")
            
            # Update subplot indices
            col_index += 1
            if col_index == 10:
                row_index += 1
                col_index = 0
                
            # Break if all subplots are filled
            if row_index == 5:
                break

    # Remove empty subplots
    for i in range(row_index, 5):
        for j in range(col_index, 10):
            fig.delaxes(axes[i, j])

    # Save the aggregated correct predictions image
    plt.tight_layout()
    plt.savefig("correct_predictions.png")