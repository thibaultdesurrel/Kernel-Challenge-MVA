import numpy as np
import pickle as pkl
import networkx as nx
from src.kernels import WL_kernel
from src.SVM import SVM
from multiprocessing import freeze_support
import argparse
import pandas as pd

#Get the arguments from the user
parser = argparse.ArgumentParser(description="Script to start the training for the 2023 data challenge - Kernel methods")
parser.add_argument(
    "--h",
    type=int,
    default=2,
    metavar="H",
    help="height of the Weisfeiler-Lehman subtree kernel",
)
parser.add_argument(
    "--C",
    type=int,
    default=10,
    metavar="C",
    help="regularization parameter C of the SVM",
)

args = parser.parse_args()
h = args.h
C = args.C


# Load the data
with open('data/training_data.pkl', 'rb') as file: train_graphs = pkl.load(file)
with open('data/test_data.pkl', 'rb') as file: test_graphs = pkl.load(file)
with open('data/training_labels.pkl', 'rb') as file: train_labels = pkl.load(file)

# Preprocess the data
## Convert the labels to {-1,1}
new_train_labels = 2*train_labels-1
train_graphs = np.array(train_graphs,dtype = type(train_graphs[0]))

if __name__ == '__main__':
    freeze_support()
    kernel = WL_kernel(h)
    print("Computing the training kernel matrix")
    K_train = kernel.kernel(train_graphs,train_graphs)
    print("Done !")

    class_weights = [len(new_train_labels) / np.sum(new_train_labels == 1), len(new_train_labels) / np.sum(new_train_labels == -1)]
    model = SVM(C, k.kernel)
    print("Fitting the model")
    model.fit(train_graphs, new_train_labels, K_train, class_weights)
    print("Model fitted")

    print("Predicting in the test dataset")
    logits = model.predict_logit(test_graphs)
    Yte = {'Predicted' : logits}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    path = "pred_h" + str(h) + "_C" + str(C) + ".csv"
    dataframe.to_csv(path,index_label='Id')
    print("Prediction done and results saved as" + path)
