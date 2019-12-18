# Detecting Pneumonia from Chest X-Rays

The project.py file implements many different machine learning algorithms to detect the presence of pneumonia in chest X-rays.
The dataset is gathered from Kaggle at the following link: (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia); 
however, the train set, test set, and validation sets that I used to train my algorithms are different than the ones provided
by Kaggle. 

In the python file, there are linear classifiers, decision tree classifiers, linear neural networks, and convolutional neural
networks. To run the specific machine learning algorithm, simply uncomment the respective code in the main() function and
ensure that the paths of the images matches where your chest X-ray images are located. Running the script (using python3) 
will show you the progress of the images being added as well as the confusion matrix numbers and accuracy, precision, recall 
numbers. 

As a brief summary, running on both validation set and test set, logistic regression reached a high of 84% accuracy, 
random forest classifier reached a high of 87% accuracy, linear neural networks achieved 88% accuracy, and convolutinoal
neural networks also achieved an 88% accuracy.
