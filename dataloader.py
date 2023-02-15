from glob import glob
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
from skimage.color import gray2rgb

class TopoDataSet(Dataset):
    def __init__(self, fileName, mode="temp", labelIdx=[6,7,8]):
        super().__init__()
        self.compose = transforms.Compose([transforms.ToTensor()])
        self.labelIdx = labelIdx
        self.fileName = glob(fileName+mode+"\\coronal\\*"+".npy")
        self.fileName2 = glob(fileName+mode+"\\sagittal\\*"+".npy")
        self.fileName3 = glob(fileName+mode+"\\transverse\\*"+".npy")

    def __getitem__(self, index):
        data = np.load(self.fileName[index])
        data2 = np.load(self.fileName2[index])
        img = gray2rgb(data[0]).astype(np.float32)

        labels = []
        boxes = []
        for i in self.labelIdx:
            mask = np.array(np.where(data[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i))
            boxes.append([x_min, y_min, x_max, y_max])

        img = self.compose(img)
        img2 = None
        img3 = None

        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = {"labels":labels, "boxes": boxes}
        return img, target
    def __len__(self):
        return len(self.fileName)

class Topo3dDataSet_20class(Dataset):
    def __init__(self, fileName, mode="temp", labelIdx=[6,7,8]):
        super().__init__()
        self.compose = transforms.Compose([transforms.ToTensor()])
        self.labelIdx = labelIdx
        self.fileName = glob(fileName+mode+"\\coronal\\*"+".npy")
        self.fileName2 = glob(fileName+mode+"\\sagittal\\*"+".npy")

    def __getitem__(self, index):

        mode = 0
        if index >= len(self.fileName):
            mode = 1
            index = index - len(self.fileName)

        data = np.load(self.fileName[index])
        data2 = np.load(self.fileName2[index])
        img = gray2rgb(data[0]).astype(np.float32)
        img2 = gray2rgb(data2[0]).astype(np.float32)

        labels = []
        boxes = []
        if mode == 0:
            for i in self.labelIdx:
                mask = np.array(np.where(data[i]==1))
                y_min,x_min = np.min(mask,axis = 1)
                y_max,x_max = np.max(mask,axis = 1)
                labels.append(int(i))
                boxes.append([x_min, y_min, x_max, y_max])
        else:
            preIDX = 10
            for i in self.labelIdx:
                mask = np.array(np.where(data2[i]==1))
                y_min,x_min = np.min(mask,axis = 1)
                y_max,x_max = np.max(mask,axis = 1)
                labels.append(int(i+preIDX))
                boxes.append([x_min, y_min, x_max, y_max])

        img = self.compose(img)
        img2 = self.compose(img2)
        img3 = None
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = {"labels":labels, "boxes": boxes}
        if mode == 0:
            return img,target
        else:
            return img2,target

    def __len__(self):
        return len(self.fileName)*2

class Topo3dDataSe_30class(Dataset):
    def __init__(self, fileName, mode="temp", labelIdx=[6,7,8]):
        super().__init__()
        self.compose = transforms.Compose([transforms.ToTensor()])
        self.labelIdx = labelIdx
        self.fileName = glob(fileName+mode+"\\coronal\\*"+".npy")
        self.fileName2 = glob(fileName+mode+"\\sagittal\\*"+".npy")
        self.fileName3 = glob(fileName+mode+"\\transverse\\*"+".npy")

    def __getitem__(self, index):
        data = np.load(self.fileName[index])
        data2 = np.load(self.fileName2[index])
        data3 = np.load(self.fileName3[index])
        img = gray2rgb(data[0]).astype(np.float32)
        img2 = gray2rgb(data2[0]).astype(np.float32)
        img3 = gray2rgb(data3[0]).astype(np.float32)
        labels = []
        boxes = []
        for i in self.labelIdx:
            mask = np.array(np.where(data[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i))
            boxes.append([x_min, y_min, x_max, y_max])

        preIDX = 10
        for i in self.labelIdx:
            mask = np.array(np.where(data2[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i+preIDX))
            boxes.append([x_min, y_min, x_max, y_max])

        for i in self.labelIdx:
            mask = np.array(np.where(data3[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i+preIDX*2))
            boxes.append([x_min, y_min, x_max, y_max])

        img = self.compose(img)
        img2 = self.compose(img2)
        img3 = self.compose(img3)
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        target = {"labels":labels, "boxes": boxes}
        return img,target

    def __len__(self):
        return len(self.fileName)

class Topo3dDataSe_6label(Dataset):
    def __init__(self, fileName, mode="temp", labelIdx=[6,7,8]):
        super().__init__()
        self.compose = transforms.Compose([transforms.ToTensor()])
        self.labelIdx = labelIdx
        self.fileName = glob(fileName+mode+"\\coronal\\*"+".npy")
        self.fileName2 = glob(fileName+mode+"\\sagittal\\*"+".npy")
        self.fileName3 = glob(fileName+mode+"\\transverse\\*"+".npy")

    def __getitem__(self, index):
        data = np.load(self.fileName[index])
        data2 = np.load(self.fileName2[index])
        data3 = np.load(self.fileName3[index])

        img = gray2rgb(data[0]).astype(np.float32)
        img2 = gray2rgb(data2[0]).astype(np.float32)
        img3 = gray2rgb(data3[0]).astype(np.float32)

        # plt.figure()
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(data[1])
        # plt.show()

        labels = []
        boxes = []
        depths = []
        for i in self.labelIdx:
            mask = np.array(np.where(data[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i))
            boxes.append([x_min, y_min, x_max, y_max])

            mask = np.array(np.where(data2[i]==1))
            z_min,_ = np.min(mask,axis = 1)
            z_max,_ = np.max(mask,axis = 1)
            depths.append([z_min, z_min,  z_max, z_max]) #  0,, 1
            # boxes.append([x_min, y_min, z_min, x_max, y_max, z_max])

        img = self.compose(img)
        # img2 = self.compose(img2)
        # img3 = self.compose(img3)
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        depths = torch.tensor(depths, dtype=torch.float32)

        target = {"labels":labels, "boxes": boxes, "depths": depths}
        return img,target

    def __len__(self):
        return len(self.fileName)

class TestDataSet(Dataset):
    def __init__(self, fileName, mode="temp", labelIdx=[6,7,8]):
        super().__init__()
        self.compose = transforms.Compose([transforms.ToTensor()])
        self.labelIdx = labelIdx
        self.fileName = glob(fileName+mode+"\\coronal\\*"+".npy")
        self.fileName2 = glob(fileName+mode+"\\sagittal\\*"+".npy")
        self.fileName3 = glob(fileName+mode+"\\transverse\\*"+".npy")

    def __getitem__(self, index):
        data = np.load(self.fileName[index])
        data2 = np.load(self.fileName2[index])
        data3 = np.load(self.fileName3[index])
        img = gray2rgb(data[0]).astype(np.float32)
        img2 = gray2rgb(data2[0]).astype(np.float32)
        img3 = gray2rgb(data3[0]).astype(np.float32)

        labels = []
        boxes = []
        for i in self.labelIdx:
            mask = np.array(np.where(data[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i))
            boxes.append([x_min, y_min, x_max, y_max])

        preIDX = 10
        for i in self.labelIdx:
            mask = np.array(np.where(data2[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i+preIDX))
            boxes.append([x_min, y_min, x_max, y_max])

        for i in self.labelIdx:
            mask = np.array(np.where(data3[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i+preIDX*2))
            boxes.append([x_min, y_min, x_max, y_max])

        img = self.compose(img)
        img2 = self.compose(img2)
        img3 = self.compose(img3)
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = {"labels":labels, "boxes": boxes}
        return [img,img2,img3], target

    def __len__(self):
        return len(self.fileName)

class TestDataSet_6label(Dataset):
    def __init__(self, fileName, mode="temp", labelIdx=[6,7,8]):
        super().__init__()
        self.compose = transforms.Compose([transforms.ToTensor()])
        self.labelIdx = labelIdx
        self.fileName = glob(fileName+mode+"\\coronal\\*"+".npy")
        self.fileName2 = glob(fileName+mode+"\\sagittal\\*"+".npy")
        self.fileName3 = glob(fileName+mode+"\\transverse\\*"+".npy")

    def __getitem__(self, index):
        data = np.load(self.fileName[index])
        data2 = np.load(self.fileName2[index])
        data3 = np.load(self.fileName3[index])
        img = gray2rgb(data[0]).astype(np.float32)
        img2 = gray2rgb(data2[0]).astype(np.float32)
        img3 = gray2rgb(data3[0]).astype(np.float32)

        labels = []
        boxes = []
        for i in self.labelIdx:
            mask = np.array(np.where(data[i]==1))
            y_min,x_min = np.min(mask,axis = 1)
            y_max,x_max = np.max(mask,axis = 1)
            labels.append(int(i))

            mask = np.array(np.where(data2[i]==1))
            z_min,_ = np.min(mask,axis = 1)
            z_max,_ = np.max(mask,axis = 1)
            boxes.append([x_min, y_min, z_min, x_max, y_max, z_max])

        img = self.compose(img)
        img2 = self.compose(img2)
        img3 = self.compose(img3)
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = {"labels":labels, "boxes": boxes}
        return [img,img2,img3], target

    def __len__(self):
        return len(self.fileName)

class TopoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, labelIdx: list = [6,7,8], mode="2d"):
        super().__init__()
        self.path = data_dir
        self.batch_size = batch_size
        self.labelIdx = labelIdx
        if mode == "2d":
            self.dataset = TopoDataSet
        elif mode == "20class":
            self.dataset = Topo3dDataSet_20class
        elif mode == "30class":
            self.dataset = Topo3dDataSe_30class
        elif mode == "6label":
            self.dataset = Topo3dDataSe_6label
        elif mode == "test":
            self.dataset = TestDataSet
        elif mode == "test_6L":
            self.dataset = TestDataSet_6label

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            _full = self.dataset(self.path,mode="train",labelIdx=self.labelIdx)
            n_data = len(_full)
            n_train = int(0.78*n_data)
            # ! tempo parameter use n_train and n_val !!!
            self._train, self._val = random_split(_full,[n_train,n_data-n_train])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self._test = self.dataset(self.path,mode="test",labelIdx=self.labelIdx)

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size)
