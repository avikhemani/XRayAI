from colorama import Fore, Back, Style
import numpy as np
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
NUM_EPOCHS = 10
BATCH_SIZE = 32
TRAIN_NORMAL_DIR = './chest_xray/train/NORMAL'
TRAIN_PNEUMONIA_DIR = './chest_xray/train/PNEUMONIA'
TEST_NORMAL_DIR = './chest_xray/val/NORMAL'
TEST_PNEUMONIA_DIR = './chest_xray/val/PNEUMONIA'

# GPU availability
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

class ConvNetwork():
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=25, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=25, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(32 * 32 * 16, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Returns np array of image matricies and corresponding classifications
def processImageData(dir, crop, normalize, canny):
    basename = os.path.basename(dir)
    pneumonia = 1 if basename == 'PNEUMONIA' else 0
    images, presence = [], []
    imagePaths = os.listdir(dir)
    length = len(imagePaths)
    for i, imagePath in enumerate(imagePaths):
        if i == 60: break
        if imagePath[0] == '.': continue
        img = cv2.imread(os.path.join(dir, imagePath), cv2.IMREAD_GRAYSCALE)
        if not crop:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            img = cv2.resize(img, (IMAGE_SIZE + 2*CROP_SIZE, IMAGE_SIZE + 2*CROP_SIZE))
            img = img[CROP_SIZE:IMAGE_SIZE+CROP_SIZE, CROP_SIZE:IMAGE_SIZE+CROP_SIZE]
        if normalize:
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if canny:
            img = cv2.Canny(img, 150, 250)
        images.append(np.asarray(img))
        presence.append(pneumonia)
        print(Fore.GREEN + "Added image " + str(i+1) + " out of " + str(length))

    return np.asarray(images), np.asarray(presence)

# Returns a concatenated list of image matricies  with classifications
def getInputOutputData(normalDir, pneumoniaDir, crop, normalize, canny):
    normalX, normalY = processImageData(normalDir, crop, normalize, canny)
    pneumoniaX, pneumoniaY = processImageData(pneumoniaDir, crop, normalize, canny)
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
    print("")
    print("Number of true positives: " + str(tp))
    print("Number of true negatives: " + str(tn))
    print("Number of false positives: " + str(fp))
    print("Number of false negative: " + str(fn))
    print("")
    print("The accuracy of this model is " + str(round(accuracy*100, 3)) + " %")
    print("The precision of this model is " + str(round(precision*100, 3)) + " %")
    print("The recall of this model is " + str(round(recall*100, 3)) + " %")

def linearRegression(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    linReg = LinearRegression()
    linReg.fit(xTrainFlat, yTrain)
    return linReg

def gradientDescentClassifier(xTrain, yTrain):
    xTrainFlat = flattenComponents(xTrain)
    sgdclas = SGDClassifier(loss='log')
    sgdclas.fit(xTrainFlat, yTrain)
    return sgdclas

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
    length = len(yTrain)
    xTrainTensors, yTrainTensors = np.array_split(xTrain, length//BATCH_SIZE), np.array_split(yTrain, length//BATCH_SIZE)
    for i in range(len(yTrainTensors)):
        xTrainTensors[i] = from_numpy(flattenComponents(xTrainTensors[i])).type(FloatTensor).to(torchDevice)
        yTrainTensors[i] = from_numpy(yTrainTensors[i]).to(torchDevice)
    nnet = NeuralNetwork().to(torchDevice)
    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.NLLLoss()
    optimizer = optim.SGD(nnet.parameters(), lr=0.001)

    nnet.train()
    for epoch in range(NUM_EPOCHS):
        for xTrainTensor, yTrainTensor in zip(xTrainTensors, yTrainTensors):
            output = nnet(xTrainTensor) # forward propogation
            loss = loss_function(output, yTrainTensor) # loss calculation
            optimizer.zero_grad()
            loss.backward() # backward propagation
            optimizer.step() # weight optimization

        print("Epoch:", epoch+1, "Training Loss: ", loss.item())

    return nnet

def convNetworkTorch(xTrain, yTrain):
    length = len(yTrain)
    xTrainTensors, yTrainTensors = np.array_split(xTrain, length//BATCH_SIZE), np.array_split(yTrain, length//BATCH_SIZE)
    for i in range(len(yTrainTensors)):
        xTrainTensors[i] = from_numpy(xTrainTensors[i]).type(FloatTensor).to(torchDevice)
        yTrainTensors[i] = from_numpy(yTrainTensors[i]).to(torchDevice)
    convnet = ConvNetwork().to(torchDevice)
    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.NLLLoss()
    optimizer = optim.SGD(convnet.parameters(), lr=0.001)

    convnet.train()
    for epoch in range(NUM_EPOCHS):
        for xTrainTensor, yTrainTensor in zip(xTrainTensors, yTrainTensors):
            output = nnet(xTrainTensor) # forward propogation
            loss = loss_function(output, yTrainTensor) # loss calculation
            optimizer.zero_grad()
            loss.backward() # backward propagation
            optimizer.step() # weight optimization

        print("Epoch:", epoch+1, "Training Loss: ", loss.item())

    return convnet

def main():
    # Gather train and test data
    xTrain, yTrain = getInputOutputData(TRAIN_NORMAL_DIR, TRAIN_PNEUMONIA_DIR, crop=False, normalize=True, canny=False)

    # Shuffle training data
    indices = np.arange(0, len(xTrain))
    np.random.shuffle(indices)
    xTrain, yTrain = xTrain[indices], yTrain[indices]

    xTest, yTest = getInputOutputData(TEST_NORMAL_DIR, TEST_PNEUMONIA_DIR, crop=False, normalize=True, canny=False)
    print("Training model...")

    # ------ LinearRegression ------
    # linReg = linearRegression(xTrain, yTrain)
    # prediction = linReg.predict(flattenComponents(xTest))
    # predClassifiers = np.asarray([1 if pred >= 0.5 else 0 for pred in prediction])
    # reportAccuracy(predClassifiers, yTest)

    # ------ SGDClassifier ------
    # sgdclas = gradientDescentClassifier(xTrain, yTrain)
    # weights = sgdclas.coef_
    # weights = np.reshape(weights, (IMAGE_SIZE, IMAGE_SIZE))
    # weights = np.add(weights, np.min(weights))
    # weights = cv2.normalize(weights, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # showImage(weights)
    # prediction = sgdclas.predict(flattenComponents(xTest))
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
    # nnet = neuralNetworkTorch(xTrain, yTrain)
    # nnet.eval()
    # xTestTensor = from_numpy(flattenComponents(xTest)).type(FloatTensor).to(torchDevice)
    # output = nnet(xTestTensor)
    # prediction_tensor = torchmax(output, 1)[1]
    # prediction = np.squeeze(prediction_tensor.cpu().numpy())
    # reportAccuracy(prediction, yTest)

    # ------ ConvNetworkTorch -----
    convnet = neuralNetworkTorch(xTrain, yTrain)
    convnet.eval()
    xTestTensor = from_numpy(flattenComponents(xTest)).type(FloatTensor).to(torchDevice)
    output = convnet(xTestTensor)
    prediction_tensor = torchmax(output, 1)[1]
    prediction = np.squeeze(prediction_tensor.cpu().numpy())
    reportAccuracy(prediction, yTest)

if __name__ == '__main__':
    main()
