import os
import pickle
import random

import cv2
import numpy as np


class Data_Preprocess_Manager:
    def create_data(self, data_dir, labels, img_size):
        data = []
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_path = os.path.join(path, img)
                    img_array = cv2.imread(img_path)
                    resized_img = cv2.resize(img_array, (img_size, img_size))
                    # plt.imshow(resized_img)
                    # plt.show()
                    print(img_path)
                    data.append([resized_img, class_num])

                except Exception as e:
                    pass

            # shuffle the training data
            random.shuffle(data)

        return data

    @staticmethod
    def create_pickle_file(training_data, x_pickle_file_name, y_pickle_file_name):
        X = []
        Y = []
        for data, label in training_data:
            X.append(data)
            Y.append(label)

        X = np.array(X)
        X = X.swapaxes(1, 3)
        print(X.shape)
        print(len(Y))

        # Create X
        print(x_pickle_file_name)
        pickle_out = open(x_pickle_file_name, "wb")
        pickle.dump(X, pickle_out, protocol=4)
        # pickle.dump(X, pickle_out)
        pickle_out.close()

        # Create Y
        print(y_pickle_file_name)
        pickle_out = open(y_pickle_file_name, "wb")
        pickle.dump(Y, pickle_out)
        pickle_out.close()


def create_pickle_for_training():
    # Texture Dataset
    # DTD
    # x_train_pickle_file = "Texture_DTD_Train_X.pickle"
    # y_train_pickle_file = "Texture_DTD_Train_Y.pickle"

    x_test_pickle_file = "Texture_DTD_Test_X.pickle"
    y_test_pickle_file = "Texture_DTD_Test_Y.pickle"

    ROOT_PATH = "./images/DTD/dtd"
    DATASET_PATH = "./Dataset/Texture/DTD"
    # TEXTURE_DATA_Train_DIR = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Dapeng_Wu/data_root_folder/dataset/texture/dtd/train_images"
    # TEXTURE_DATA_Test_DIR = "/Users/shantanughosh/Desktop/Shantanu_MS/Research/Dapen_Wu/data_root_folder/dataset/texture/dtd/test_images"
    TEXTURE_LABELS = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
                      "crosshatched", "crystalline",
                      "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed",
                      "interlaced", "knitted",
                      "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted", "pleated",
                      "polka-dotted", "porous",
                      "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
                      "studded", "swirly", "veined",
                      "waffled", "woven", "wrinkled", "zigzagged"]

    dpm = Data_Preprocess_Manager()

    # texture data preprocess
    # testing
    for i in range(10):
        idx = i + 1
        TEXTURE_DATA_Test_DIR = ROOT_PATH + "/test" + str(idx)
        print(TEXTURE_DATA_Test_DIR)
        texture_training_data = dpm.create_data(TEXTURE_DATA_Test_DIR, TEXTURE_LABELS, 227)
        x_train_pickle_file = DATASET_PATH + "/Texture_DTD_test" + str(idx) + "_X.pickle"
        y_train_pickle_file = DATASET_PATH + "/Texture_DTD_test" + str(idx) + "_Y.pickle"
        dpm.create_pickle_file(texture_training_data, x_train_pickle_file, y_train_pickle_file)

    # training
    for i in range(10):
        idx = i + 1
        TEXTURE_DATA_Test_DIR = ROOT_PATH + "/train" + str(idx)
        print(TEXTURE_DATA_Test_DIR)
        texture_training_data = dpm.create_data(TEXTURE_DATA_Test_DIR, TEXTURE_LABELS, 227)
        x_train_pickle_file = DATASET_PATH + "/Texture_DTD_train" + str(idx) + "_X.pickle"
        y_train_pickle_file = DATASET_PATH + "/Texture_DTD_train" + str(idx) + "_Y.pickle"
        dpm.create_pickle_file(texture_training_data, x_train_pickle_file, y_train_pickle_file)

    # validation
    for i in range(10):
        idx = i + 1
        TEXTURE_DATA_Test_DIR = ROOT_PATH + "/val" + str(idx)
        print(TEXTURE_DATA_Test_DIR)
        texture_training_data = dpm.create_data(TEXTURE_DATA_Test_DIR, TEXTURE_LABELS, 227)
        x_train_pickle_file = DATASET_PATH + "/Texture_DTD_val" + str(idx) + "_X.pickle"
        y_train_pickle_file = DATASET_PATH + "/Texture_DTD_val" + str(idx) + "_Y.pickle"
        dpm.create_pickle_file(texture_training_data, x_train_pickle_file, y_train_pickle_file)


create_pickle_for_training()
