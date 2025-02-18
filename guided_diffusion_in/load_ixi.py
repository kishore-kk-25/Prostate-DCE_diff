import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd,Rotate90
)
from monai.data import Dataset
import h5py
import threading


def load_data(ixi_path,h5_path):
    ixi_path = ixi_path + '/**_Guys/T2/NIfTI/**'
    transforms_1 = Compose(
    [
     AddChanneld(('image')),
     Orientationd(('image'),'RAS'),
     ScaleIntensityD(('image',)),
     ToTensord(('image')),
    ])
    
    dataset = H5CachedDataset(ixi_path,transforms_1,h5cachedir=h5_path)

    return dataset

class H5CachedDataset(Dataset):
    def __init__(self, datapath,
                 transforms_1, 
                 nslices_per_image = 130 ,
                 start_slice = 60,
                 end_slice = 45,
                 h5cachedir=None):

        if h5cachedir is not None:
            if not os.path.exists(h5cachedir):
                os.mkdir(h5cachedir)
            self.cachedir = h5cachedir
        niilist=[]
        for x in glob.glob(datapath):
            niilist.append(x)

        self.datalist = niilist

        self.xfms = transforms_1
                    
        #### 3d image loader from monai
        
        self.loader = LoadImage()
        self.loader.register(NibabelReader())
        
        #### start_slice & end_slice---> slices to be truncated in each volume vol[:,:,start_slice:-end_slice]
        
        self.start_slice = start_slice
        self.end_slice = end_slice
        
        self.nslices = nslices_per_image - self.end_slice

        
    def __len__(self):
        #### total number of slices in all the volumes
        return len(self.datalist)*(self.nslices - self.start_slice)
    
    def clear_cache(self):
        #### function to clear the directory storing h5 files (used for caching the h5 files)
        for fn in os.listdir(self.cachedir):
            os.remove(self.cachedir+'/'+fn)
            
    def __getitem__(self,index):
        #### ditionary to store data slicewise
        data = {}
        label ={}
#      
#        
        filenum = index // (self.nslices - self.start_slice)


        slicenum = index % (self.nslices - self.start_slice)

        slicenum += self.start_slice
        
        #### Extract the datafile location & mask file location based on filenum

        datalist_filenum = self.datalist[filenum]

        loc_data = datalist_filenum

            
        ##### if h5 exists for the current volume fill data dictionary with current slice number
        if self.cachedir is not None:
            h5name = self.cachedir+'/%d.h5' % filenum

            ptname = self.cachedir+'/%d.pt' % filenum

            if os.path.exists(h5name):
                
                with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
                    for key in itm.keys():                       
                        data[key]=torch.from_numpy(itm[key][:,:,:,slicenum])



        ##### if data dictionary is empty #######
        if len(data)==0:
            
            imgdata, meta = self.loader(loc_data)

            ### Rotating it ###
            imgdata = torch.rot90(imgdata,3)
            
            #### store volume wise image & mask data,metadata in a dictionary 
            data_i = {'image':imgdata}

            #### transform the data dictionary
            data3d = self.xfms(data_i)

            if self.cachedir is not None:
                other = {}

                with h5py.File(h5name,'w',libver='latest') as itm:
                    itm.swmr_mode = True
                    for key in data3d:
                        if key in ['image']:                             
                            img_npy = data3d[key].numpy()

                            shp = img_npy.shape
                            
                            chunk_size = list(shp[:-1])+[1]
                            ds = itm.create_dataset(key,shp,chunks=tuple(chunk_size),dtype=img_npy.dtype)
                            ds[:]=img_npy[:]
                            ds.flush()
                    else:
                        other[key]=data3d[key]
                torch.save(other,ptname)
                
            data = {'image':data3d['image'][:,:,:,slicenum]}

            
        if len(data)>0:

            res = data
            res['filenum'] = filenum
            res['slicenum'] = slicenum
            res['idx'] = index
            return res['image'] , {"y":np.array((1))}

        else:
            # replace with random
            return self.__getitem__(np.random.randint(len(self.datalist)))
