############################
# Author: Reza Katebi
# This code loads the dataset
# and divides it to 80 % training
# and 20 % testing data. It
# also makes a batch loader using
# pytorch's dataloader
###########################
import numpy as np
import torch
# from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


class DataLoad:
    """
    Loads the dataset and divides it to train and
    test and makes them Tensordataset and loaders
    for loading batches.
    """

    def __init__(self, batch_size):
        X = np.load('../data/train_downsample.npy')[:61500]
        y = np.load('../data/train_labels_regression.npy')[:61500]
        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train)

        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    Galaxy = DataLoad(100)
    a = Galaxy.train_loader
    for batch_id, (data, target) in enumerate(a):

        target = torch.eye(5).index_select(dim=0, index=target)
        print(data.shape)
        print(target.shape)
