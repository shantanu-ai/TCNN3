import torch


class Util:
    @staticmethod
    def convert_to_tensor(X, Y, device):
        """
        Converts the dataset to tensor.

        :param X: dataset
        :param Y: label
        :param device: whether {cpu or gpu}

        :return: the dataset as tensor
        """
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_y = torch.from_numpy(Y)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset

    @staticmethod
    def convert_to_tensor_test(X):
        """
        Converts the dataset to tensor.

        :param X: dataset
        :param Y: label
        :param device: whether {cpu or gpu}

        :return: the dataset as tensor
        """
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        processed_dataset = torch.utils.data.TensorDataset(tensor_x)
        return processed_dataset

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
