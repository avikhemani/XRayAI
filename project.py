from colorama import Fore, Back, Style
import numpy as np
#import tensorflow as tf
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from torch import nn, optim, tensor, from_numpy, FloatTensor, device, cuda, max as torchmax
import torch.nn.functional as F
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
torchDevice = device('cuda' if cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(IMAGE_SIZE**2, 4000)
        nn.init.xavier_uniform_(self.hidden1.weight)
        # self.hidden2 = nn.Linear(4000, 200)
        # nn.init.xavier_uniform_(self.hidden2.weight)
        self.output = nn.Linear(4000, 2)
        nn.init.xavier_uniform_(self.output.weight)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        # x = self.hidden2(x)
        # x = self.relu(x)
        x = self.output(x)
        x = F.softmax(x, dim=1)
        return x

# Returns np array of image matricies and corresponding classifications
def processImageData(dir, crop, normalize):
    basename = os.path.basename(dir)
    pneumonia = 1 if basename == 'PNEUMONIA' else 0
    images, presence = [], []
    imagePaths = os.listdir(dir)
    length = len(imagePaths)
    for i, imagePath in enumerate(imagePaths):
        if pneumonia == 1 and i == 2000: break
        if imagePath[0] == '.': continue
        img = cv2.imread(os.path.join(dir, imagePath), cv2.IMREAD_GRAYSCALE)
        if not crop:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            img = cv2.resize(img, (IMAGE_SIZE + 2*CROP_SIZE, IMAGE_SIZE + 2*CROP_SIZE))
            img = img[CROP_SIZE:IMAGE_SIZE+CROP_SIZE, CROP_SIZE:IMAGE_SIZE+CROP_SIZE]
        if normalize:
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images.append(np.asarray(img))
        presence.append(pneumonia)
        print(Fore.GREEN + "Added image " + str(i+1) + " out of " + str(length))

    return np.asarray(images), np.asarray(presence)

# Returns a concatenated list of image matricies  with classifications
def getInputOutputData(normalDir, pneumoniaDir, crop, normalize):
    normalX, normalY = processImageData(normalDir, crop, normalize)
    pneumoniaX, pneumoniaY = processImageData(pneumoniaDir, crop, normalize)
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
    xTrainTensor = from_numpy(flattenComponents(xTrain)).type(FloatTensor).to(torchDevice)
    yTrainTensor = from_numpy(yTrain).to(torchDevice)
    nnet = NeuralNetwork().to(torchDevice)
    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.NLLLoss()
    optimizer = optim.SGD(nnet.parameters(), lr=0.001)

    nnet.train()
    for epoch in range(NUM_EPOCHS):
        output = nnet(xTrainTensor) # forward propogation
        loss = loss_function(output, yTrainTensor) # loss calculation
        optimizer.zero_grad()
        loss.backward() # backward propagation
        optimizer.step() # weight optimization

        # nnet.eval()
        # output = nnet(xTrainTensor)
        # loss = loss_function(output, yTrainTensor)
        # valid_loss.append(loss.item())
        # print("Epoch:", epoch+1, "Training Loss: ", np.mean(train_loss), "Validation Loss: ", np.mean(valid_loss))
        print("Epoch:", epoch+1, "Training Loss: ", loss.item())

    return nnet

def main():
    # Gather train and test data
    xTrain, yTrain = getInputOutputData(TRAIN_NORMAL_DIR, TRAIN_PNEUMONIA_DIR, crop=False, normalize=True)
    xTest, yTest = getInputOutputData(TEST_NORMAL_DIR, TEST_PNEUMONIA_DIR, crop=False, normalize=True)
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
    xTestTensor = from_numpy(flattenComponents(xTest)).type(FloatTensor).to(torchDevice)
    output = nnet(xTestTensor)
    prediction_tensor = torchmax(output, 1)[1]
    prediction = np.squeeze(prediction_tensor.cpu().numpy())
    reportAccuracy(prediction, yTest)
    print(prediction)
    print(yTest)

if __name__ == '__main__':
    main()
