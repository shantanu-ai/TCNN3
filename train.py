import copy

import torch.utils.data
from torch import nn, optim

from MTLCNN_single import MTLCNN_single
from Util import Util
from dataLoader import DataLoader


def main():
    TEXTURE_LABELS = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
                      "crosshatched", "crystalline",
                      "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed",
                      "interlaced", "knitted",
                      "lacelike", "lined", "marbled", "matted", "meshed", "paisley", "perforated", "pitted", "pleated",
                      "polka-dotted", "porous",
                      "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
                      "studded", "swirly", "veined",
                      "waffled", "woven", "wrinkled", "zigzagged"]

    print("Texture_label: " + str(len(TEXTURE_LABELS)))
    model_path_bn = "./Models/Auto_encoder_Model_epoch_300_lr_0.001_noise_factor_0.5.pt"

    device = Util.get_device()
    print(device)

    train_parameters = {
        "epochs": 400,
        "learning_rate": 0.0001,
        # "learning_rate": 0.0005,
        "texture_batch_size": 32,
        "image_net_batch_size": 256
    }

    texture_train_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_train{0}_X.pickle"
    texture_train_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_train{0}_Y.pickle"

    texture_val_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_val{0}_X.pickle"
    texture_val_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_val{0}_Y.pickle"

    saved_model_name = "./Models/Texture_Single_Classifier_Model_epoch_" + str(
        train_parameters["epochs"]) + "_lr_" + str(
        train_parameters["learning_rate"]) + "_split{0}.pth"

    # training started
    data_loader_list = prepare_data_loader_train_10_splits(texture_train_data_set_path, texture_train_label_set_path,
                                                           texture_val_data_set_path, texture_val_label_set_path,
                                                           32, 0, device)
    train_arguments = {
        "TEXTURE_LABELS": TEXTURE_LABELS,
        "data_loader_list": data_loader_list,
        "train_parameters": train_parameters,
        "saved_model_name": saved_model_name
    }

    network = train(train_arguments, device)

    print('Saved model parameters to disk.')
    # training ended


def prepare_data_loader_train_10_splits(texture_train_data_set_path, texture_train_label_set_path,
                                        texture_val_data_set_path, texture_val_label_set_path,
                                        texture_batch_size, num_workers, device):
    data_loader_list = []
    for i in range(10):
        idx = i + 1
        print("Split: {0}".format(idx))
        texture_train_data_set_path = texture_train_data_set_path.format(idx)
        texture_train_label_set_path = texture_train_label_set_path.format(idx)
        texture_val_data_set_path = texture_val_data_set_path.format(idx)
        texture_val_label_set_path = texture_val_label_set_path.format(idx)

        dL = DataLoader()
        texture_train_set, train_set_size = dL.get_tensor_set(texture_train_data_set_path,
                                                              texture_train_label_set_path,
                                                              device)
        texture_val_set, val_set_size = dL.get_tensor_set(texture_val_data_set_path,
                                                          texture_val_label_set_path,
                                                          device)
        print("Train set size: {0}".format(train_set_size))
        print("Val set size: {0}".format(val_set_size))

        texture_train_data_loader = torch.utils.data.DataLoader(texture_train_set,
                                                                batch_size=texture_batch_size,
                                                                shuffle=True,
                                                                num_workers=num_workers)
        texture_val_data_loader = torch.utils.data.DataLoader(
            texture_val_set, num_workers=1, shuffle=False, pin_memory=True)

        data_loader_dict = {
            "train": texture_train_data_loader,
            "val": texture_val_data_loader
        }
        data_loader_list.append(data_loader_dict)

    return data_loader_list


def train(train_arguments, device):
    TEXTURE_LABELS = train_arguments["TEXTURE_LABELS"]
    data_loader_list = train_arguments["data_loader_list"]
    train_parameters = train_arguments["train_parameters"]
    saved_model_name = train_arguments["saved_model_name"]

    print("..Training started..")
    epochs = train_parameters["epochs"]
    lr = train_parameters["learning_rate"]
    phases = ['train', 'val']
    # set batch size

    # set optimizer - Adam

    split_id = 0

    # start training
    for data_loader_dict in data_loader_list:
        # initialise network for each dataset
        network = MTLCNN_single(TEXTURE_LABELS).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()
        min_correct = 0
        split_id += 1
        print('-' * 50)
        print("Split: {0} =======>".format(split_id))

        # start epoch
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 20)

            for phase in phases:
                if phase == 'train':
                    network.train()  # Set model to training mode
                else:
                    network.eval()  # Set model to evaluate mode

                running_loss = 0
                running_correct = 0
                total_image_per_epoch = 0

                for batch in data_loader_dict[phase]:
                    images, label = batch
                    images = images.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()

                    output = network(images)
                    loss = criterion(output, label).to(device)
                    total_image_per_epoch += images.size(0)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * images.size(0) * 2
                    running_correct += get_num_correct(output, label)

                epoch_loss = running_loss / total_image_per_epoch

                epoch_accuracy = running_correct / total_image_per_epoch
                print(
                    "{0} ==> loss: {1}, correct: {2}/{3}, accuracy: {4}".format(phase, epoch_loss, running_correct,
                                                                                total_image_per_epoch,
                                                                                epoch_accuracy))
                if phase == 'val' and running_correct > min_correct:
                    print("saving model with correct: {0}, improved over previous {1}"
                          .format(running_correct, min_correct))
                    min_correct = running_correct
                    best_model_wts = copy.deepcopy(network.state_dict())
                    torch.save(best_model_wts, saved_model_name.format(split_id))

    return network


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


main()
