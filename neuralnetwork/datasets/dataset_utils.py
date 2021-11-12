from importlib import resources
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

def OneOfKEncode(df):
    newdf = df.applymap(str)
    newdf = pd.get_dummies(newdf, prefix=df.columns)
    newdf = newdf.applymap(float)
    return newdf

def rescale(labels, neg):  # Rescaling from 0-1 to (neg) - (1 - neg)
    mul = 1 - (2 * neg)
    labels = 2 * labels
    labels = labels - 1

    labels = mul * labels

    labels = labels + 1
    labels = labels / 2

    return labels

def load_monk_dataset(n):
    neg_label = 0.01
    path = './neuralnetwork/datasets/monk{}'.format(n)
    pathTrain = path+ '/train'
    pathTest = path +'/test'

    # Read dataset
    train_examples = pd.read_csv(pathTrain, sep=" ", usecols=[2, 3, 4, 5, 6, 7])
    train_examples = OneOfKEncode(train_examples)
    train_labels = pd.read_csv(pathTrain, sep=" ", usecols=[1])
    train_labels = rescale(train_labels, neg_label)

    # Read dataset
    test_examples = pd.read_csv(pathTest, sep=" ", usecols=[2, 3, 4, 5, 6, 7])
    test_examples = OneOfKEncode(test_examples)
    test_labels = pd.read_csv(pathTest, sep=" ", usecols=[1])

    return np.array(train_examples), np.array(train_labels), np.array(test_examples), np.array(test_labels)


def load_cup():
    randomSeed = 420000
    dataPath = './neuralnetwork/datasets/cup/ML-CUP19-TR.csv'

    # Read dataset
    X = pd.read_csv(dataPath, skiprows=7, header=None, usecols=list(range(1, 21)))
    y = pd.read_csv(dataPath, skiprows=7, header=None, usecols=[21, 22])

    X = np.array(X)
    y = np.array(y)

    X, y = shuffle(X, y, random_state=randomSeed)

    return X,  y
