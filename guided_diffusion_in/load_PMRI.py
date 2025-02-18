import os
import glob
import numpy as np
import torch
import pathlib
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


class SliceData(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    "Loader hsa been changed for 70 percent train,20 percent val,10 percent for test"
    
    
    def __init__(self, root, transforms1=None, transforms2=None, noise_level=0, mode='train', train_test_split = 0.8):
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        files = sorted([str(i) for i in files])
#         #print('files',files[:-3])
        imgs=[]
        transforms = Compose([AddChanneld(('T2','bval0','bval188','ADC','DCE_01','DCE_02','DCE_03')),\
                              Orientationd(('T2','bval0','bval188','ADC','DCE_01','DCE_02','DCE_03'),'RAS'),\
                              Resized(keys = ('T2','bval0','bval188','ADC','DCE_01','DCE_02','DCE_03'),spatial_size = (128, 128,-1),\
                                      mode = 'trilinear',\
                                      align_corners = True)])
        # if(noise_level == 0):
        #     self.xfms = transforms2
        # else :
        #     self.xfms = transforms1
        self.xfms = transforms
        

        if mode == "train":
            for filename in files[:int(train_test_split * len(files))]:
                if filename[-3:] == '.h5':
                    imgs.append(filename)
                    
        # elif mode=='val':
        #     for filename in files[int(train_test_split * len(files)):int((train_test_split+0.1) * len(files))]:
        #         if filename[-3:] == '.h5':
        #             imgs.append(filename)
            
                    
                    
        elif mode == 'test':
            for filename in files[int((train_test_split) * len(files)):]:
                if filename[-3:] == '.h5':
                    imgs.append(filename)
#         #print('len of imgs',len(imgs))

        self.examples = []


        for fname in imgs[:]:
            with h5py.File(fname,'r') as hf:
                fsvol = hf['T2']
                num_slices = fsvol.shape[-1]
                mid = num_slices//2
#                 #print("mid",mid)
                self.examples += [(fname, slice) for slice in range(mid-6,mid+6)] #2 to 14 
            

    def __len__(self):
        # #print('len ',len(self.examples))
        return len(self.examples)
    def normalize(self,source):
        source = (source-np.min(source))/(np.max(source) - np.min(source))
        return source

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
#         #print(fname,slice)
        final_dic ={}
        with h5py.File(fname, 'r') as data:    
            t2 = data['T2'][0,:,:,:].astype(np.float64)
            adc = data['ADC'][0,:,:,:].astype(np.float64)
            
            dwi0 = data['bval0'][0,:,:,:].astype(np.float64)
            dwi188 = data['bval188'][0,:,:,:].astype(np.float64)
            
            dce_01 = data['DCE_01'][0,:,:,:].astype(np.float64)
            dce_02 = data['DCE_02'][0,:,:,:].astype(np.float64)
            dce_03 = data['DCE_03'][0,:,:,:].astype(np.float64)
            mask = data['mask'][0,:,:,:].astype(np.float64)
        #print("after h5",t2.shape)
        
        dic = {"T2": t2,"ADC":adc,
                "bval0": dwi0,'bval188':dwi188,"DCE_01":dce_01,
                "DCE_02": dce_02, "DCE_03":dce_03,'mask':mask}#,'ktrans':ktrans}

        # if self.xfms:
        trans_dic = self.xfms(dic)
        #print("transdic",trans_dic['T2'].shape)
        
        t2 = trans_dic['T2'][:,:,:,slice].astype(np.float64)
        adc = trans_dic['ADC'][:,:,:,slice].astype(np.float64)
        dwi0 = trans_dic['bval0'][:,:,:,slice].astype(np.float64)
        dwi188  =trans_dic['bval188'][:,:,:,slice].astype(np.float64)
        dce_01 = trans_dic['DCE_01'][:,:,:,slice].astype(np.float64)
        dce_02 = trans_dic['DCE_02'][:,:,:,slice].astype(np.float64)
        dce_03 = trans_dic['DCE_03'][:,:,:,slice].astype(np.float64)
#         else:
#             trans_dic = dic
#             t2 = trans_dic['T2'][np.newaxis,:,:,slice].astype(np.float64)
#             adc = trans_dic['ADC'][np.newaxis,:,:,slice].astype(np.float64)
#             pd = trans_dic['PD'][np.newaxis,:,:,slice].astype(np.float64)
#             dce_01 = trans_dic['DCE_01'][np.newaxis,:,:,slice].astype(np.float64)
#             dce_02 = trans_dic['DCE_02'][np.newaxis,:,:,slice].astype(np.float64)
#             dce_03 = trans_dic['DCE_03'][np.newaxis,:,:,slice].astype(np.float64)
        
    
    
    
        # trans_dic  =dic
        #print("transdic1",t2.shape)
        # t2 = trans_dic['T2'][np.newaxis,:,:].astype(np.float64)
        # #print("after new axis",t2.shape)
        # adc = trans_dic['ADC'][np.newaxis,:,:].astype(np.float64)
        # dwi0  =trans_dic['bval0'][np.newaxis,:,:].astype(np.float64)
        # dwi188  =trans_dic['bval188'][np.newaxis,:,:].astype(np.float64)
        # dce_01 = trans_dic['DCE_01'][np.newaxis,:,:].astype(np.float64)
        # dce_02 = trans_dic['DCE_02'][np.newaxis,:,:].astype(np.float64)
        # dce_03 = trans_dic['DCE_03'][np.newaxis,:,:].astype(np.float64)

        # t2 = t2[np.newaxis,:,:].astype(np.float64)
        # #print("after new axis",t2.shape)
        # adc = adc[np.newaxis,:,:].astype(np.float64)
        # dwi0  =dwi0[np.newaxis,:,:].astype(np.float64)
        # dwi188  =dwi188[np.newaxis,:,:].astype(np.float64)
        # dce_01 = dce_01[np.newaxis,:,:].astype(np.float64)
        # dce_02 = dce_02[np.newaxis,:,:].astype(np.float64)
        # dce_03 = dce_03[np.newaxis,:,:].astype(np.float64)
        # mask = trans_dic['mask'][np.newaxis,:,:,slice].astype(np.float64)
         
        t2 =  torch.from_numpy(t2)#self.normalize(t2))
        adc =  torch.from_numpy(adc)#self.normalize(adc))
        dwi0=  torch.from_numpy(dwi0)#self.normalize(dwi0))
        dwi188 = torch.from_numpy(dwi188)
        
        dce_01=  torch.from_numpy(dce_01)#self.normalize(dce_01))
        dce_02 =  torch.from_numpy(dce_02)#self.normalize(dce_02))
        dce_03 =  torch.from_numpy(dce_03)#self.normalize(dce_03))
        # mask =  torch.from_numpy(mask)
        ##print("t2 shape befroe return ",t2.shape)
        # resvit_one = {'A':torch.cat((t2,dwi0,dwi188,adc,dce_01),axis=0),'B':mask}
        
        # batch = torch.cat((t2,dwi0,adc,dce_01,dce_02,dce_03),axis=0)
        
        batch = torch.cat((t2,t2,adc,dce_01,dce_02,dce_03),axis=0)
        
        return batch,{"y":np.array((1))}

        # resvit_one = {'A':torch.cat((t2,dwi0,dwi188,adc,dce_01),axis=0),'B':torch.cat((dce_02,dce_03),axis=0}
        

#         #print(resvit_one.keys())
        
#         dic = {"T2": torch.from_numpy(t2),"ADC":torch.from_numpy(adc),
#                 "PD": torch.from_numpy(pd), "DCE_01":torch.from_numpy(dce_01),
#                 "DCE_02": torch.from_numpy(dce_02), "DCE_03":torch.from_numpy(dce_03)}
        
        # return resvit_one#dic