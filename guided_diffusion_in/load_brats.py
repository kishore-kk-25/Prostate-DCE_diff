import os
import glob
import numpy as np
import torch
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import (
    Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
    # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)
from monai.data import Dataset
import h5py
import threading


def load_data(brats_path,h5_path):
    brats_path = brats_path + '/**/**_t1.nii'
    transforms_1 = Compose(
    [
     AddChanneld(('image')),
     Orientationd(('image'),'RAS'),
     ScaleIntensityD(('image',)),
     ToTensord(('image')),
    ])
    
    dataset = H5CachedDataset(brats_path,transforms_1,h5cachedir=h5_path)
    
    #### preparing dataloader#####
    
#     brats_loader = torch.utils.data.DataLoader(dataset,batch_size = batch_size ,\
#                                                shuffle = True ,num_workers= 1)
    
    return dataset
    
    

modality_list = ['t1', 't2']
def comb_path(path):
    dict_ = {}
    for itr_ in modality_list:
        uniq_ = path.split('/')[-1].split('_')
        basepath = "/".join(path.split('/')[0:-1])
        b = '_'.join(uniq_[0:-1])
        c = '/'.join([basepath,b])
        itr__ = itr_+'.nii'
        d ='_'.join([c,itr__])
        dict_.update({itr_:d})
        
    return dict_

class H5CachedDataset(Dataset):
    def __init__(self, datapath,
                 transforms_1, transforms_2= ToTensord(('image',)),
                 nslices_per_image = 155 ,
                 start_slice = 10,
                 end_slice = 10,
                 h5cachedir=None):
       #         self.lock = threading.Lock()
        if h5cachedir is not None:
            if not os.path.exists(h5cachedir):
                os.mkdir(h5cachedir)
            self.cachedir = h5cachedir
        niilist=[]
        for x in glob.glob(datapath):
            t1,t2 = comb_path(x)['t1'],comb_path(x)['t2']
            niilist.append(t1)
            niilist.append(t2)

        self.datalist = niilist

        
        self.xfms = transforms_1
        self.xfms2 = transforms_2
        
            
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

        l = loc_data.split('/')[-1].split('_')[-1]
        if( l == 't1.nii'):
            label["y"] = np.array((0))
        if( l == 't2.nii'):
            label["y"] = np.array((1))
            
        ##### if h5 exists for the current volume fill data dictionary with current slice number
        if self.cachedir is not None:
            h5name = self.cachedir+'/%d.h5' % filenum

            ptname = self.cachedir+'/%d.pt' % filenum

            if os.path.exists(h5name):
                
                with h5py.File(h5name,'r',libver='latest', swmr=True) as itm:
                    for key in itm.keys():                       
                        data[key]=torch.from_numpy(itm[key][:,:,:,slicenum])
                data['image_meta_dict']={'affine':np.eye(3)} # FIXME: loses spacing info - can read from pt file


        ##### if data dictionary is empty
        if len(data)==0:
            
            imgdata, meta = self.loader(loc_data)
            imgdata_padded = torch.zeros((256,256,155))
            imgdata_padded[8:-8,8:-8,:] = imgdata

            
            #### store volume wise image & mask data,metadata in a dictionary 
            data_i = {'image':imgdata_padded}

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
#                             print(chunk_size)
                            ds = itm.create_dataset(key,shp,chunks=tuple(chunk_size),dtype=img_npy.dtype)
                            ds[:]=img_npy[:]
#                             print('ds shape',ds.shape)
                            ds.flush()
                    else:
                        other[key]=data3d[key]
                torch.save(other,ptname)
                
            data = {'image':data3d['image'][:,:,:,slicenum],              
                'image_meta_dict':{
                    'affine':np.eye(3)}}

            
        if len(data)>0:
#             print("**",data.keys())
#             res = self.xfms2(data)
            res = data
#             print(res.keys())
#             res['image'] = res['image'].float()
            res['filenum'] = filenum
            res['slicenum'] = slicenum
            res['idx'] = index
            return res['image'] , label

        else:
            # replace with random
            return self.__getitem__(np.random.randint(len(self.datalist)))
