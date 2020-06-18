# Crack-detection-

 

AI Model to detect the crack defect
18.06.2020
─
Deepika Tiwari

Overview
We use deep learning to build a simple yet very accurate model for crack detection.Furthermore, we test the model on real world data and see that the model is accurate in detecting  cracks.
Goals
Build an AI model to accurately detect the crack defect.
Develop model, deploy on a computer.
Make the model work on a mobile phone.
Data Set
Using the data given - 
Data set consists of 111 defected images and 139 healthy images.
Load and preprocess our data:
To load our data, we grab all paths to images in the dataset_dir directory . Then, for each imagePath, we:
Extract the class label (either defected or healthy) from the path
Load the image, and preprocess it by converting to RGB channel ordering, and resizing it to 224×224 pixels so that it is ready for our Convolutional Neural Network.
Update our data and labels lists respectively.
We then scale pixel intensities to the range [0, 1] and convert both our data and labels to NumPy array format.
Next we will one-hot encode our labels and create our training/testing splits :
One-hot encoding of labels takes place meaning that our data will be in the following format: [[0. 1.] [0. 1.] [0. 1.] … [1. 0.] [1. 0.] [1. 0.]]
Each encoded label consists of a two element array with one of the elements being “hot” (i.e., 1) versus “not” (i.e., 0).
Then construct our data split, reserving 80% of the data for training and 20% for testing.
In order to ensure that our model generalizes, we perform data augmentation by setting the random image rotation setting to 15 degrees clockwise or counterclockwise. We will Initialize the data augmentation generator object.

We will initialize our VGGNet model and set it up for fine-tuning :
We will instantiate the VGG16 network with weights pre-trained on ImageNet, leaving off the FC layer head.
From there, we construct a new fully-connected layer head consisting of POOL => FC = SOFTMAX layers and append it on top of VGG16.
We then freeze the CONV weights of VGG16 such that only the FC layer head will be trained; this completes our fine-tuning setup.

Compile and train our deep learning model:
Compile the network with learning rate decay and the Adam optimizer. Given that this is a 2-class problem, we use “binary_crossentropy” loss rather than categorical crossentropy.
To kick off our neural network training process, we make a call to Keras’ fit_generator method, while passing in our  data via our data augmentation object.
Evaluate Model:
For evaluation, we first make predictions on the testing set and grab the prediction indices.
We then generate and print out a classification report using scikit-learn’s helper utility.
Compute a confusion matrix for further statistical evaluation:
Here we will:
Generate a confusion matrix
Use the confusion matrix to derive the accuracy, sensitivity, and specificity and print each of these metrics
Plot our training:
We plot our training accuracy/loss history for inspection, outputting the plot to an image file:Finally we serialize our tf.keras classifier model to disk.





















