import os
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from PIL import ImageOps
import pandas as pd
import numpy as np

class ImageTestset(Dataset):
    def __init__(self,list_IDs,filenames,
                img_dir='/results/bea/Data/siim-isic-melanoma-classification/jpeg/test',
                max_resolution=512,
                transform=transforms.ToTensor()):
        Dataset.__init__(self)
        self.transform = transform
        self.data_dir=img_dir
        self.max_res=max_resolution,max_resolution
        self.res=max_resolution
        self.IDs=list_IDs
        self.filenames=filenames

    def __getitem__(self,idx):
        image_name=self.filenames[idx]
        filename=str(image_name)+'.jpg'
        image=Image.open(os.path.join(self.data_dir,filename))
        image_resize=image.resize(self.max_res)
        array=(np.array(image_resize)-128)/256
        X=np.float32(np.reshape(array,(3,self.res,self.res)))
        return X,image_name
    
    def __len__(self):
        return len(self.IDs)
