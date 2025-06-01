
# Required methods from scikit-learn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Required Libraries
import numpy as np
import matplotlib.pyplot as plt

# Used for confusion metrix data
from sklearn import metrics 

## Start Program Message
print("\n__________________________________________________\n")
print(" Machine learning app that classifies handwritten digits (0–9)\n")


## Loading digits from sklearn.datasets
print("\t[!]-> Loading data...\n")
digits = load_digits()
if (digits):
    print("\t[!]-> Data loaded successfully!\n")
    print("\t - Data Shape: ", digits.data.shape)
    print("\t - Data Size: ", digits.data.size)
    print()
else:
    print("\t[X]-> Error when loading data")
    print("\n")
    SystemExit

## Normalizing Data, [0, 16] -> [0, 1]
print("\t[!]-> Normalizing Data...\n")
digits.data = digits.data/16.0
print("\t[!]-> Data normalized successfully!\n")

## Displaying digits data
print("\t[!]-> Displaying Data...\n")
input = int(input("\t[i]-> Enter number of images to preview: "))
print()

# Conditionally Display plot
if (input != 0):

    # Setting dimesions for column/row for diplaying digits images dynamically inside plot
    columns = 10
    rows = (input + columns-1) // columns
    plt.figure(figsize=(columns*2,rows*2.2))

    # Traversing digits data and creating pairs of each image with its corresponing label and adds it in subplot to display
    for index, (image, label) in enumerate(zip(digits.data[0:input], digits.target[0:input])):
        plt.subplot(rows, columns, index+1)
        plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
        plt.title('%i\n' % label, fontsize = 20)
        plt.axis('off')
    
    # Displays Plot 
    plt.tight_layout()
    plt.show()
    print("\n\t[!]-> Data displayed successfully!\n")

## Splitting Data into Training Set and Test Set
print("\t[!]-> Splitting Data for Training and Tests (80/20)...\n")
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=0)
print("\t - Image Train Shape: ", x_train.data.shape)
print("\t - Labels Train Shape: ", y_train.data.shape)
print("\t - Image Test Shape: ", x_test.data.shape)
print("\t - Labels Test Shape: ", y_test.data.shape)
print("\n\t[!]-> Data splitted successfully!\n")

## scikit-learn Modeling Pattern
print("\t[!]-> Choosing Data Modelling Pattern...\n")

# Importing the Logistic Regression model from sklearn library
from sklearn.linear_model import LogisticRegression

# Creating instance of Model
logisticRegression = LogisticRegression()
print("\t - ", logisticRegression)
print("\n\t[!]-> Data Modelling Pattern set successfully!\n")


## Training Model with the training data
print("\t[!]-> Training Model using training data...\n")
logisticRegression.fit(x_train, y_train)
print("\t[!]-> Model trained successfully!\n")

## Predicting the labes from test images
print("\t[!]-> Making predictions on entire test dataset...\n")
predictions = logisticRegression.predict(x_test) # Predicting y_test value
print("\t[!]-> Predictions made successfully!\n")

## Measuring Model Performance
print("\t[!]-> Measuring Model Performance...\n")
score = logisticRegression.score(x_test, y_test)
print("\t - Score of Model Performance: %", round(score*100, 2))
print("\n\t[!]-> Model performance measured succesfully!\n")

# Function used to create and populate confusion matrix plot
print("\t[!]-> Displaying Confusion Matrix...\n")
def plot_confusion_matrix(confusionMatrix, title="Confusion Matrix", cmap="Pastel1"):
    plt.figure(figsize=(9,9))
    plt.imshow(confusionMatrix, interpolation="nearest", cmap=cmap)
    plt.title(title, size=15)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xlabel('Predicted Label', size=15)
    plt.xticks(tick_marks, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], size=10)
    plt.ylabel('Actual Label', size=15)
    plt.yticks(tick_marks, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], size=10)
    plt.tight_layout()
    width, height = confusionMatrix.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(confusionMatrix[x][y]), xy=(y,x), horizontalalignment='center', verticalalignment='center')

## Plotting and Displaying Confussion Matrix
confusionMatrix = metrics.confusion_matrix(y_test, predictions)
plot_confusion_matrix(confusionMatrix)
plt.show()
print("\n\t[!]-> Confussion Matrix generated successfully!\n")

## Displaying Classification Report
print("\t[!]-> Displaying Classification Report...\n")
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
print("\n\t[!]-> Classification Report generated successfully!\n")


