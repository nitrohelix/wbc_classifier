# White Blood Cell Classification CNN

## Introduction

This project develops a Convolutional Neural Network (CNN) model to classify images of white blood cells. The goal is to create a tool that can assist medical laboratory technicians in the manual differential process, particularly for abnormal patient results, by providing a preliminary classification of white blood cells.

**DISCLAIMER**: This model is intended for informational purposes only and should not be considered a substitute for professional medical advice or the final judgment of a qualified medical professional. Patient confidentiality is secured, and data is handled with strict privacy protocols.

## Project Structure

The project is structured as follows:

1.  **Unzip the dataset and import modules**: Extracts the image dataset from a zip file and imports necessary Python libraries.
2.  **Load and preprocess the data**: Loads the images and labels, preprocesses them (resizing, normalization, one-hot encoding), and splits the data into training and testing sets.
3.  **Build the CNN model**: Defines the architecture of the CNN model using TensorFlow/Keras.
4.  **Train the model**: Trains the compiled CNN model using the training data.
5.  **Evaluate the model**: Evaluates the trained model's performance on the test set.

## Dataset

The dataset used in this project is `raabin_wbc.zip`, containing images of white blood cells. The images are expected to be organized in subdirectories within the extracted folder, such as `Train`, `TestA`, and `TestB`. The filename format is assumed to be 'label\_imageid.jpg', where 'label' represents the type of white blood cell.

## Model Architecture

The CNN model architecture consists of:

-   Convolutional layers with ReLU activation.
-   Max-pooling layers for down-sampling.
-   A Flatten layer to convert the 2D feature maps into a 1D vector.
-   A Dense layer with ReLU activation.
-   A Dropout layer for regularization.
-   A final Dense output layer with Softmax activation for multi-class classification.

## Results

The model was trained for 15 epochs with a batch size of 64.

-   **Test Loss**: Approximately 0.0648
-   **Test Accuracy**: Approximately 0.9815

These results indicate that the model performs well on the white blood cell classification task.

## Insights and Future Work

The project successfully demonstrates the feasibility of using a CNN for white blood cell classification with high accuracy. Future work could involve:

-   Data augmentation to increase the size and diversity of the training dataset.
-   Hyperparameter tuning to optimize the model's performance.
-   Experimenting with different CNN architectures.
-   Visualizing model predictions and analyzing misclassifications to identify areas for improvement.
-   Integrating the model into a user-friendly application for practical use in a laboratory setting.

## Conclusion

This project serves as a proof of concept for applying deep learning to automate the classification of white blood cells. The high accuracy achieved suggests that intelligent microscopy has the potential to significantly reduce the workload of medical laboratory technicians and improve efficiency in the diagnostic process.
