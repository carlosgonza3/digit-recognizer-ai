PROGRAM_NAME = "MNISTClassifier"

import random


# Required Libraries
import numpy as np
import matplotlib.pyplot as plt

# Used for confusion metrix data
from sklearn import metrics 

# Used for loading MNIST Data
from struct import unpack

import sys
import termios

def flush_input():
    termios.tcflush(sys.stdin, termios.TCIFLUSH)

# Function used for debugging
def debugCommand(type, output=None, name=None):
    if (type=='task'):
        print('\n\t# -> ', output, '...\n')
    elif (type=='success'):
        print('\t! ->', output+'!\n\n')
    elif (type=='error'):
        print('\tX -> Error performing: ', output, '\n')
        SystemExit(2)
    elif (type=='start'):
        print('\n@ -> Program', PROGRAM_NAME,'running...\n')
    elif (type=='end'):
        print('\n@ -> Program', PROGRAM_NAME,'finished executing\n')
    elif (type=='data'):
        print('\t\t* -> ', name, ': ', output,'\n')
    else:
        print(output)

# Function that loads MNIST dataset from ./data
def loadmnist(imageFile, labelFile):
    
    # Open the images with gzip in read binary mode
    images = open(imageFile, 'rb')
    labels = open(labelFile, 'rb')

    images.read(4)
    numberOfImages = images.read(4)  # Skip the magic_number for the fixed header identifier
    numberOfImages = unpack('>I', numberOfImages)[0] 

    rows = images.read(4)
    rows = unpack('>I', rows)[0]

    columns = images.read(4)
    columns = unpack('>I', columns)[0]

    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]

    x = np.zeros((N, rows*columns), dtype=np.uint8) # Initializing nump array
    y = np.zeros(N, dtype=np.uint8) # Initializing nump array

    for i in range(N):
        for j in range(rows*columns):
            tmp_pixel = images.read(1)
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            x[i][j] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    
    images.close() # Prevent data leaks
    labels.close() # Prevent data leaks
    
    return (x, y)

# Start Program
debugCommand('start')

# Extracting training data (img/label) from the MNIST dataset
debugCommand('task', 'Extracting training data from MNIST dataset')
train_img, train_lbl = loadmnist('data/train-images-idx3-ubyte', 'data/train-labels-idx1-ubyte')
debugCommand('data', train_img.shape, 'train_img shape')
debugCommand('data', train_lbl.shape, 'train_lbl shape')
debugCommand('success', 'Training data saved successfully')

# Extracting learning data (img/label) from the MNIST datase
debugCommand('task', 'Extracting test data from MNIST dataset')
test_img, test_lbl = loadmnist('data/t10k-images-idx3-ubyte', 'data/t10k-labels-idx1-ubyte')
debugCommand('data', test_img.shape, 'test_img shape')
debugCommand('data', test_lbl.shape, 'test_lbl shape')
debugCommand('success', 'Test data saved successfully')

# Displaying training Images
count = int(input('\t_ -> # of images to load: '))
print()
cols = 10
rows = (count + cols - 1) // cols  # Ceiling division

plt.figure(figsize=(cols * 2, rows * 2.2))  # Dynamic figsize

for index, (image, label) in enumerate(zip(test_img[:count], test_lbl[:count])):
    plt.subplot(rows, cols, index + 1)
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.title(f'{label}', fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Normalizing Data
debugCommand("task", "Normalizing data...")

debugCommand('data', '', "Normalizing train_img")
train_img = train_img/255
#print("\n", train_img[0], "\n")

#print("\n", train_lbl, "\n")

debugCommand('data', '', "Normalizing test_img")
test_img = test_img/255
#print("\n", test_img[0], "\n")

debugCommand('success', "Data is now normalized")
#print("\n", test_lbl, "\n")


# Logistic Linear Regression
debugCommand("task", "Setting learning model...")
from sklearn.linear_model import LogisticRegression
debugCommand('data', LogisticRegression, "Learning Model")
logisticRegression = LogisticRegression(solver='lbfgs', max_iter=1000)
debugCommand('success', "Learning model set")

# Training Data
debugCommand("task", " Training Model...")
logisticRegression.fit(train_img, train_lbl)
debugCommand("success", " Model is trained")

def testModel():

    testIndex = ""

    while testIndex != -1:

        flush_input()
        testIndex = input('\t_ -> Test index (-1 exit):')
        print()

        if (testIndex == ""):
            testIndex = random.randint(0, len(test_img) - 1)
            debugCommand('data', testIndex, 'Index selected')
        else:
            testIndex = int(testIndex)

        if testIndex == -1:
            break

        show_digit_image(test_img[testIndex], test_lbl[testIndex])

        prediction = logisticRegression.predict(test_img[testIndex].reshape(1, -1))[0]
        actual = test_lbl[testIndex]
        debugCommand('data', f"Predicted: {prediction}", f"Actual: {actual}")

        testIndex = ""

def show_digit_image(image, label):
    plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.title(f'{label}', fontsize=10)
    plt.axis('off')
    plt.show()

debugCommand("task", " Testing Model...")
testModel()
debugCommand("success", " Model Tested")

# Measuring Model Performance
score = logisticRegression.score(test_img, test_lbl)
debugCommand('data', (round(score*100, 2),'%',' of accuracy'), 'Model Performance')

# Measuring Model Performance
debugCommand("task", " Generating Predictions")
predictions = logisticRegression.predict(test_img)
debugCommand("success", " Predictions Generated")

# Displaying confusion matrix
debugCommand("task", " Ploting confusion matrix")
confusionMatrix = metrics.confusion_matrix(test_lbl, predictions)
def plot_confusion_matrix(confusionMatrix, title="Confusion Matrix", cmap="Pastel1"):
    plt.figure(figsize=(9,9))
    plt.imshow(confusionMatrix, interpolation="nearest", cmap=cmap)
    plt.title(title, size=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xlabel('Predicted Label', size=15)
    plt.xticks(tick_marks, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], size=10)
    plt.xlabel('Actual Label', size=15)
    plt.yticks(tick_marks, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], size=10)
    plt.tight_layout()
    width, height = confusionMatrix.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(confusionMatrix[x][y]), xy=(y,x), horizontalalignment='center', verticalalignment='center')
plot_confusion_matrix(confusionMatrix)
plt.show()
debugCommand("success", " Cconfusion matrix displayed")

## Displaying Classification Report
debugCommand('task', 'Generating classification report')
from sklearn.metrics import classification_report
print(classification_report(test_lbl, predictions), '\n')
debugCommand('success', 'classification report created')

debugCommand('end')