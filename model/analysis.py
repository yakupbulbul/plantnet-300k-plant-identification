import os
import pandas as pd
import numpy as np
import json
from utils import *
import config as cfg

def dataOperation(model_list,createCsv = False):

    if createCsv:


        create_df(str(cfg.IMAGES_TRAIN_DIR), str(cfg.NAMES_JSON), save=True)

    fe = FeatureExtraction()

    fe._getModelAndFuse(model_list)
    fe._indexAllData()


def search(modelList,queryImage,nImg):

    search = SearchByIndexFile()
    data = pd.read_pickle(cfg.FEATURES)
    extracted = search._extractQuery(queryImage,modelList)
    dicti = search._searchByIndex(extracted,nImg,data)
    return dicti







def create_df(root_path: os.path,metadata:os.path,save : bool= False):
    
    names = sorted(os.listdir(root_path))

    path_list = []
    name_list = []
    label_list = []

    with open(metadata,"r") as f:

        loaded = json.load(f)
    f.close()


    

    count = 0
    for name in names:

        images = os.path.join(root_path,name)
        


        for img in sorted(os.listdir(images)):

            absolute_path = os.path.join(images, img)
            path_list.append(os.path.relpath(absolute_path, cfg.DATASET_ROOT))
        
            name_list.append(loaded[os.path.basename(images)])
            label_list.append(count)

        count+=1

    

    df = pd.DataFrame()

    df["NAMES"] = name_list
    df["PATHS"] = path_list
    df["LABELS"] = label_list

    if save :
        df.to_csv("dataWithImages.csv",index=False)
    return df

if __name__=="__main__":

    # dataOperation(cfg.MODELS)

    sample_image = os.getenv("PLANTNET_SAMPLE_IMAGE")
    if sample_image:
        print(
            search(
                cfg.MODELS,
                sample_image,
                cfg.WILL_RETURN_IMAGE_COUNT,
            )
        )
    else:
        print("Set PLANTNET_SAMPLE_IMAGE to run a quick search test.")


            
            


    
    
