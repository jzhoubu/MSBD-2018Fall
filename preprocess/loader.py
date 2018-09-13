import os,sys,tqdm
import numpy as np



class loader(object):
    def __init__(self, PATH):
        self.path=PATH
        resize_flag=False

    def resize(self,shape=None):
        if len(shape)!=3:
            print("resize shape should be a list or array with shape (3, )")
        self.resize_flag=shape

    def load_image_iter(self,image_files):
        for v in tqdm(np.asarray(self.image_path).reshape(-1)):
            if resize_flag:
                yield resize(image=imread(v),output_shape=self.resize_shape,mode='constant')
            else:
                yield imread(v)
    
    def load_image(self):
        files=os.listdir(self.path)
        self.labels=list([name[0:2]for name in files]) # Suppose label is the first 2 filename
        self.image_path = list([os.path.join(self.path,file) for file in files])
        self.images=np.asarray(list(self.load_image_iter()))


    # def display(self, sample=5, label=None): 
    # 未必能plot
