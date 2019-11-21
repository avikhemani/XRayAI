from colorama import Fore, Back, Style
import numpy as np
#import tensorflow as tf
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from torch import nn, optim, tensor, from_numpy, FloatTensor, max as torchmax
import cv2
import os

# Constants
IMAGE_SIZE = 200
CROP_SIZE = 50
NUM_EPOCHS = 50
TRAIN_NORMAL_DIR = './chest_xray/train/NORMAL'
TRAIN_PNEUMONIA_DIR = './chest_xray/train/PNEUMONIA'
TEST_NORMAL_DIR = './chest_xray/val/NORMAL'
TEST_PNEUMONIA_DIR = './chest_xray/val/PNEUMONIA'

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(IMAGE_SIZE**2, 8000)
        self.sigmoid = nn.Sigmoid()
        # self.hidden2 = nn.Linear(4000, 200)
        # self.relu = nn.ReLU()
        self.output = nn.Linear(8000, 2)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.sigmoid(x)
        # x = self.hidden2(x)
        # x = self.relu(x)
        x = self.output(x)
        return x

# Returns np array of image matricies and corresponding classifications
def processImageData(dir, crop):
    basename = os.path.basename(dir)
    pneumonia = 1 if basename == 'PNEUMONIA' else 0
    images, presence = [], []
    imagePaths = os.listdir(dir)
    length = len(imagePaths)
    for i, imagePath in enumerate(imagePaths):
        if imagePath[0] == '.': continue
        img = cv2.imread(os.path.join(dir, imagePath), cv2.IMREAD_GRAYSCALE)
        if not crop:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            img = cv2.resize(img, (IMAGE_SIZE + 2*CROP_SIZE, IMAGE_SIZE + 2*CROP_SIZE))
            img = img[CROP_SIZE:IMAGE_SIZE+CROP_SIZE, CROP_SIZE:IMAGE_SIZE+CROP_SIZE]
        images.append(np.asarray(img))
        presence.append(pneumonia)
        print(Fore.GREEN + "Added image " + str(i+1) + " out of " + str(length))

    return np.asarray(images), np.asarray(presence)

# Returns a concatenated list of image matricies  with classifications
def getInputOutputData(normalDir, pneumoniaDir, crop=False):
    normalX, normalY = processImageData(normalDir, crop)
    pneumoniaX, pneumoniaY = processImageData(pneumoniaDir, crop)
    xTrain = np.concatenate((normalX, pneumoniaX))
    yTrain = np.concatenate((normalY, pneumoniaY))
    return xTrain, yTrain

# Flattens the matrices within the array parameter
def flattenComponents(array):
    N, H, W = array.shape
    return np.reshape(array, (N, H*W))

def showImage(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reportAccuracy(prediction, actual):
    print(Style.RESET_ALL)
    assert len(prediction) == len(actual)

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(prediction)):
        predVal = prediction[i]
        actVal = actual[i]
        if predVal == 1 and actVal == 1: tp += 1
        if predVal == 1 and actVal == 0: fp += 1
        if predVal == 0 and actVal == 1: fn += 1
        if predVal == 0 and actVal == 0: tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("Training completed.")
    print("The accuracy of this model is " + str(round(accuracy*100, 3)) + " %")
    print("The precision of this model is " + str(round(precision*100, 3)) + " %")
    print("The recall of this model is " + str(round(recall*100, 3)) + " %")

def linearRegression(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    linReg = LinearRegression()
    linReg.fit(xTrainFlat, yTrain)
    return linReg

def logisticRegression(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    logReg = SGDClassifier(loss='log', learning_rate='optimal', eta0=0.01)
    logReg.fit(xTrainFlat, yTrain)
    return logReg

def naiveBayes(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    navBay = GaussianNB()
    navBay.fit(xTrainFlat, yTrain)
    return navBay

def decisionTreeClassifier(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    decTree = DecisionTreeClassifier()
    decTree.fit(xTrainFlat, yTrain)
    return decTree

def randomForestClassifier(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    randFor = RandomForestClassifier(criterion='gini', n_estimators=50)
    randFor.fit(xTrainFlat, yTrain)
    return randFor

def supportVectorClassifier(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    svc = SVC()
    svc.fit(xTrainFlat, yTrain)
    return svc

def neuralNetworkSK(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    nn = MLPClassifier(activation='logistic',solver='sag')
    nn.fit(xTrainFlat, yTrain)
    return nn

def neuralNetworkTorch(xTrain, yTrain):
    xTrainTensor = from_numpy(flattenComponents(xTrain)).type(FloatTensor)
    yTrainTensor = from_numpy(yTrain)
    nnet = NeuralNetwork()
    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.NLLLoss()
    optimizer = optim.SGD(nnet.parameters(), lr=0.01)

    train_loss = []
    nnet.train()
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        output = nnet(xTrainTensor) # forward propogation
        loss = loss_function(output, yTrainTensor) # loss calculation
        loss.backward() # backward propagation
        optimizer.step() # weight optimization
        train_loss.append(loss.item())

        # nnet.eval()
        # output = nnet(xTrainTensor)
        # loss = loss_function(output, yTrainTensor)
        # valid_loss.append(loss.item())
        # print("Epoch:", epoch+1, "Training Loss: ", np.mean(train_loss), "Validation Loss: ", np.mean(valid_loss))
        print("Epoch:", epoch+1, "Training Loss: ", np.mean(train_loss))

    return nnet

def main():
    # Gather train and test data
    xTrain, yTrain = getInputOutputData(TRAIN_NORMAL_DIR, TRAIN_PNEUMONIA_DIR, crop=False)
    xTest, yTest = getInputOutputData(TEST_NORMAL_DIR, TEST_PNEUMONIA_DIR, crop=False)
    print("Training model...")

    # ------ LinearRegression ------
    # linReg = linearRegression(xTrain, yTrain)
    # prediction = linReg.predict(flattenComponents(xTest))
    # predClassifiers = np.asarray([1 if pred >= 0.5 else 0 for pred in prediction])
    # reportAccuracy(predClassifiers, yTest)

    # ------ LogisticRegression ------
    # logReg = logisticRegression(xTrain, yTrain)
    # prediction = logReg.predict(flattenComponents(xTest))
    # reportAccuracy(prediction, yTest)

    # ------ NaiveBayes ------
    # navBay = naiveBayes(xTrain, yTrain)
    # prediction = navBay.predict(flattenComponents(xTest))
    # reportAccuracy(prediction, yTest)

    # ------ DecisionTreeClassifier -----
    # decTree = decisionTreeClassifier(xTrain, yTrain)
    # prediction = decTree.predict(flattenComponents(xTest))
    # reportAccuracy(prediction, yTest)

    # ------ RandomForestClassifier -----
    # randFor = randomForestClassifier(xTrain, yTrain)
    # prediction = randFor.predict(flattenComponents(xTest))
    # reportAccuracy(prediction, yTest)

    # ------ SupportVectorClassifier -----
    # svc = supportVectorClassifier(xTrain, yTrain)
    # prediction = svc.predict(flattenComponents(xTest))
    # reportAccuracy(prediction, yTest)

    # ------ NeuralNetworkSK -----
    # nn = neuralNetworkSK(xTrain, yTrain)
    # prediction = nn.predict(flattenComponents(xTest))
    # reportAccuracy(prediction, yTest)

    # ------ NeuralNetworkTorch -----
    nnet = neuralNetworkTorch(xTrain, yTrain)
    nnet.eval()
    output = nnet(from_numpy(flattenComponents(xTest)).type(FloatTensor))
    prediction_tensor = torchmax(output, 1)[1]
    prediction = np.squeeze(prediction_tensor.numpy())
    reportAccuracy(prediction, yTest)

if __name__ == '__main__':
    main()
