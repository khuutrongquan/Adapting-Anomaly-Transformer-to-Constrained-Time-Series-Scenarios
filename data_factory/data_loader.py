from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

class ECGSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load dữ liệu train và scale
        train_data = np.load(f"{data_path}/ECG_train.npy") 
        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)

        # Load dữ liệu test và scale
        test_data = np.load(f"{data_path}/ECG_test.npy")
        self.test = self.scaler.transform(test_data)

        # Load label riêng cho test
        self.test_labels = np.load(f"{data_path}/ECG_test_label.npy")

        # Tạo validation set
        self.val = self.train[int(len(self.train) * 0.8):]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = self.train[index:index + self.win_size]
            return np.float32(x), np.zeros_like(x)  # Không cần label khi train

        elif self.mode == "val":
            x = self.val[index:index + self.win_size]
            return np.float32(x), np.zeros_like(x)

        elif self.mode == "test":
            x = self.test[index:index + self.win_size]
            label = self.test_labels[index + self.win_size - 1]  # Lấy label cuối của window
            label = np.repeat(label, x.shape[1])  # Nếu model dùng toàn bộ input
            return np.float32(x), np.float32(label)

        else:  # prediction mode
            x = self.test[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]
            label = self.test_labels[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size]
            return np.float32(x), np.float32(label)

class UCRSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/UCR_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/UCR_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/UCR_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class Gesture2DSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        print(data_path + "/2DGesture_train.npy")
        data = np.load(data_path + "/2DGesture_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/2DGesture_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/2DGesture_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

def get_loader_segment(data_path, batch_size, win_size=100, step=1, mode='train', dataset='ECG'):
    if (dataset == 'ECG'):
        dataset = ECGSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'UCR'):
        dataset = UCRSegLoader(data_path, win_size, step, mode)
    elif (dataset == '2DGesture'):
        dataset = Gesture2DSegLoader(data_path, win_size, step, mode)
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
