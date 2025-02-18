import os
import glob
import numpy as np
import torch
import pathlib
from tqdm.notebook import tqdm
from monai.data import ITKReader,NibabelReader
from monai.transforms import LoadImage, LoadImaged
from monai.transforms import Orientationd, EnsureChannelFirstd, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd,AddChanneld
from monai.data import Dataset
import h5py
import threading


class SliceData(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """


    def __init__(self, root, mode='train', train_test_split = 0.8):

        files = list(pathlib.Path(root).iterdir())
        files = sorted([str(i) for i in files])
        imgs=[]
        # transforms = Compose([AddChanneld(('T2','PD','ADC','DCE_01','DCE_02','DCE_03')),\
        #                       Orientationd(('T2','PD','ADC','DCE_01','DCE_02','DCE_03'),'RAS'),\
        #                       Resized(keys = ('T2','PD','ADC','DCE_01','DCE_02','DCE_03'),spatial_size = (128, 128,-1),\
        #                               mode = 'trilinear',\
        #                               align_corners = True)])
        # FOR KTRANS
        
        transforms = Compose([AddChanneld(('T2','PD','ADC','DCE_01','DCE_02','DCE_03','KTRANS')),\
                              Orientationd(('T2','PD','ADC','DCE_01','DCE_02','DCE_03','KTRANS'),'RAS'),\
                              Resized(keys = ('T2','PD','ADC','DCE_01','DCE_02','DCE_03','KTRANS'),spatial_size = (128, 128,-1),\
                                      mode = 'trilinear',\
                                      align_corners = True)])
        
        
        self.xfms = transforms
        if mode == "train":
            for filename in files[:int(train_test_split * len(files))]:

                if filename[-3:] == '.h5':

                    imgs.append(filename)
                    
        elif mode == 'test' or 'validation':
            for filename in files[int(train_test_split * len(files)):]:
                if filename[-3:] == '.h5':
                    imgs.append(filename)


        self.examples = []


        for fname in imgs:
            with h5py.File(fname,'r') as hf:
                fsvol = hf['T2']
                num_slices = fsvol.shape[-1]
                self.examples += [(fname, slice) for slice in range(num_slices)]
            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
        final_dic ={}
        
        with h5py.File(fname, 'r') as data:
            t2 = torch.from_numpy(data['T2'][0,:,:,:].astype(np.float64))
            adc = torch.from_numpy(data['ADC'][0,:,:,:].astype(np.float64))
            pd = torch.from_numpy(data['PD'][0,:,:,:].astype(np.float64))
            dce_01 = torch.from_numpy(data['DCE_01'][0,:,:,:].astype(np.float64))
            dce_02 = torch.from_numpy(data['DCE_02'][0,:,:,:].astype(np.float64))
            dce_03 = torch.from_numpy(data['DCE_03'][0,:,:,:].astype(np.float64))
            ktrans = torch.from_numpy(data['KTRANS'][0,:,:,:].astype(np.float64))
            
        dict_ = {"T2": t2,"ADC":adc,
                "PD": pd, "DCE_01":dce_01,
                "DCE_02": dce_02, "DCE_03":dce_03,"KTRANS":ktrans}
        
        trans_dic = self.xfms(dict_) # 

        t2 = trans_dic['T2'][:,:,:,slice]
        adc = trans_dic['ADC'][:,:,:,slice]
        pd = trans_dic['PD'][:,:,:,slice]
        dce_01 = trans_dic['DCE_01'][:,:,:,slice]
        dce_02 = trans_dic['DCE_02'][:,:,:,slice]
        dce_03 = trans_dic['DCE_03'][:,:,:,slice]
        ktrans = trans_dic['KTRANS'][:,:,:,slice]
        ktrans = torch.fliplr(torch.flipud(ktrans))
        # ktrans = k_trans[np.newaxis,:,:]

        # batch = torch.cat((t2,pd,adc,dce_01,dce_02,dce_03))
        
        # batch = torch.cat((t2,pd,adc,dce_01,dce_02,dce_03)) #  this  is for without ADC experiment
        batch = torch.cat((t2,pd,adc,dce_01,dce_02,dce_03,ktrans)) 
        
        # batch = torch.cat((t2,pd,adc,dce_01,dce_02))

#         data_lst = {'input_contrast':torch.cat((t2,pd,dce_01),axis=0),\
#                     'target_contrast':torch.cat((dce_02,dce_03),axis=0)} data_lst['input_contrast'],data_lst['target_contrast'],
        
        return batch,{"y":np.array((1))}