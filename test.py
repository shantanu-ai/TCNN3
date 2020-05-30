import numpy as np
import torch.utils.data

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

    device = Util.get_device()
    print(device)

    texture_data_set_path = "./Dataset/Texture/DTD/Texture_DTD_test{0}_X.pickle"
    texture_label_set_path = "./Dataset/Texture/DTD/Texture_DTD_test{0}_Y.pickle"

    data_loader_test_list = prepare_data_loader_test_10_splits(texture_data_set_path, texture_label_set_path,
                                                               device)

    model_path_bn = "./Models/Texture_Single_Classifier_Model_epoch_400_lr_0.0001_split{0}.pth"

    test_arguments = {
        "data_loader_test_list": data_loader_test_list,
        "model_path_bn": model_path_bn,
        "TEXTURE_LABELS": TEXTURE_LABELS
    }
    test(test_arguments, device)


def prepare_data_loader_test_10_splits(texture_test_data_set_path, texture_test_label_set_path,
                                       device):
    data_loader_list = []
    for i in range(10):
        idx = i + 1
        print("Split: {0}".format(idx))
        texture_test_data_set_path = texture_test_data_set_path.format(idx)
        texture_test_label_set_path = texture_test_label_set_path.format(idx)

        dL = DataLoader()
        texture_test_set, test_set_size = dL.get_tensor_set(texture_test_data_set_path,
                                                            texture_test_label_set_path,
                                                            device)
        print("Test set size: {0}".format(test_set_size))

        test_data_loader = torch.utils.data.DataLoader(texture_test_set, num_workers=1, shuffle=False, pin_memory=True)

        data_loader_list.append(test_data_loader)

    return data_loader_list


def test(test_parameters, device):
    data_loader_test_list = test_parameters["data_loader_test_list"]
    model_path_bn = test_parameters["model_path_bn"]
    TEXTURE_LABELS = test_parameters["TEXTURE_LABELS"]

    print(model_path_bn)
    print(device)
    print("..Testing started..")

    split_id = 0
    accuracy_list = []

    # start testing
    for data_loader in data_loader_test_list:
        split_id += 1

        print('-' * 50)
        print("Split: {0} =======>".format(split_id))
        model_path = model_path_bn.format(split_id)
        print("Model: {0}".format(model_path))
        network_model = MTLCNN_single(TEXTURE_LABELS).to(device)
        network_model.load_state_dict(torch.load(model_path, map_location=device))
        network_model.eval()
        total_image_per_epoch = 0
        texture_corrects = 0

        for batch in data_loader:
            images, label = batch
            images = images.to(device)
            label = label.to(device)

            outputs = network_model(images)
            total_image_per_epoch += images.size(0)
            texture_corrects += get_num_correct(outputs, label)

        texture_corrects_accuracy = texture_corrects / total_image_per_epoch
        accuracy_list.append(texture_corrects_accuracy)
        print("total:{0} texture accuracy: {1}".format(texture_corrects, texture_corrects_accuracy))

    accuracy_np = np.asarray(accuracy_list)
    print("Mean accuracy: {0}".format(np.mean(accuracy_np)))


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


main()
