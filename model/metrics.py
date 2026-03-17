import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import config as cfg
import pandas as pd
from glob import glob
from tqdm import tqdm


import torchvision.transforms as T

from api import *

class Data(Dataset):
    def __init__(self,root_path,transforms,labels):
        super().__init__()

        self.rd = os.path.abspath(root_path)
        self.transforms = transforms
        self.labels = labels.copy()

        def to_abs_path(p):
            return (
                p
                if os.path.isabs(p)
                else os.path.abspath(os.path.join(cfg.DATASET_ROOT, p))
            )

        if "PATHS_ABS" not in self.labels.columns:
            self.labels["PATHS_ABS"] = self.labels["PATHS"].apply(to_abs_path)

        img_folders = sorted(glob(f"{self.rd}/*"))
        self.img_list = []
        for img_fold in img_folders:
            self.img_list.extend(sorted(glob(f"{img_fold}/*")))

        

        for imgs in self.img_list:
            try:
                img = cv.imread(imgs)
                img = cv.resize(img,(224,224),interpolation=cv.INTER_AREA)
            except cv.error:
                self.img_list.remove(imgs)


        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):


        img = cv.imread(self.img_list[index])
        img = cv.resize(img,(224,224),interpolation=cv.INTER_AREA)
        transformed = self.transforms(img)
        label = self.labels[self.labels["PATHS_ABS"] == self.img_list[index]]["LABELS"].values[-1]
        
        return transformed,label


def test(dataLoader):
    predicted_list = []
    true_label_list = []
    for i,(img,labels) in tqdm(enumerate(list(dataLoader)),total=len(list(dataLoader)),colour="blue"):

        dicti = search(["resnet50","efficientnet_v2_s"],img,5)
        if type(dicti) is str:
            predicted_list.append(labels.item()-1)
        else:
            predicted_list.append(dicti["foundedImage"][0])
        true_label_list.append(labels.item())

    for i in range(len(predicted_list)):
        if predicted_list[i] == None:
        
            predicted_list[i] =0
    
    print("accuracy: ",accuracy_score(true_label_list,predicted_list))
    print("precision: ",precision_score(true_label_list,predicted_list,average="weighted"))

    print("recall: ",recall_score(true_label_list,predicted_list,average="weighted"))
        

    print("f1_score: ",f1_score(true_label_list,predicted_list,average="weighted"))
    

    return true_label_list,predicted_list


if __name__ == "__main__":
    data = pd.read_csv(cfg.IMAGES_PATH_DF)
    labels = data["LABELS"].unique().tolist()
    dataset = Data(str(cfg.IMAGES_TEST_DIR), cfg.TRANSFORMS, data)
    dataLoader = DataLoader(dataset)
    print(len(list(dataLoader)))
    test(dataLoader)











