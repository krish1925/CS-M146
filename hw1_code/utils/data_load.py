import os
import numpy as np


def load(filename) :
    """
    Load csv file into X array of features and y array of labels.
    Parameters
    --------------------
    filename -- string, filename
    """
    # determine filename
    dir = os.path.dirname(__file__)
    f = os.path.join(dir,'../', filename)
    # load data
    with open(f, 'r') as fid :
        data = np.loadtxt(fid, delimiter=",")
    # separate features and labels
    X = data[:,:-1]
    y = data[:,-1]
    return X,y 

