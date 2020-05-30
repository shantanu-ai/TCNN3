import os

import numpy as np
import sklearn.model_selection as sklearn

from Util import Util


class DataLoader:
    def __init__(self):
        self.texture_set_size = 0

    def pre_process_test(self, data_set_path):
        test_data = self.__read_dataset_test(data_set_path)
        processed_dataset = Util.convert_to_tensor_test(test_data)
        return processed_dataset

    def pre_process_test_texture(self, data_set_path, label_set_path, device):
        test_data, labels_set = self.__read_dataset(data_set_path, label_set_path)
        processed_dataset = Util.convert_to_tensor(test_data, labels_set, device)
        self.texture_set_size = labels_set.shape[0]
        return processed_dataset

    def get_tensor_set(self, dataset_path, label_set_path,
                       device):
        train_data_set, labels_set = self.__read_dataset(dataset_path, label_set_path)
        texture_data_set_size = labels_set.shape[0]
        train_set = Util.convert_to_tensor(train_data_set, labels_set, device)

        return train_set, texture_data_set_size

    def get_texture_set_size(self):
        return self.texture_set_size

    def __split_train_test_validation_set_texture(self, data_set_path, label_set_path, split_size,
                                                  device):
        print("Texture Dataset Size")
        train_data_set, labels_set = self.__read_dataset(data_set_path, label_set_path)
        self.texture_set_size = labels_set.shape[0]
        X_train, X_val, Y_train, Y_val = self.__spilt_data_set(train_data_set,
                                                               labels_set,
                                                               split_size=split_size)
        train_set = Util.convert_to_tensor(X_train, Y_train, device)
        val_set = Util.convert_to_tensor(X_val, Y_val, device)
        return train_set, val_set, train_data_set, labels_set

    def __read_dataset_test(self, data_set_path):
        """
        Reads the dataset.

        :param data_set_path:
        :param label_set_path:

        :return: dataset
        """
        root_path = os.path.abspath(os.path.dirname(__file__))
        data_set_path = os.path.join(root_path, data_set_path)

        data_set = np.load(data_set_path, allow_pickle=True)
        test_data = np.array(data_set)
        test_data = test_data / 255

        print(data_set.shape)
        return test_data

    def __read_dataset(self, data_set_path, label_set_path):
        """
        Reads the dataset.

        :param data_set_path:
        :param label_set_path:

        :return: dataset
        """
        root_path = os.path.abspath(os.path.dirname(__file__))
        data_set_path = os.path.join(root_path, data_set_path)
        label_set_path = os.path.join(root_path, label_set_path)

        data_set = np.load(data_set_path, allow_pickle=True)
        labels = np.load(label_set_path, allow_pickle=True)
        train_data = np.array(data_set)
        train_data = train_data / 255
        long_labels = np.array(labels, dtype=np.longlong)

        print(data_set.shape)
        print(long_labels.shape)
        return train_data, long_labels

    @staticmethod
    def __spilt_data_set(data_set, label_set, split_size):
        """
        Splits the data set into test and train set.

        :param data_set: dataset
        :param label_set: true labels
        :param split_size: split percentage

        :return: train and test dataset and corresponding labels
        """
        X_train, X_test, Y_train, Y_test = \
            sklearn.train_test_split(data_set, label_set, test_size=split_size, stratify=label_set)

        return X_train, X_test, Y_train, Y_test
