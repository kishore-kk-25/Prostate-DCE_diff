import os
import glob
import numpy as np
import torch
import pathlib
import matplotlib.pyplot as plt
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
from skimage import exposure



class SliceData(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    "Loader hsa been changed for 70 percent train,20 percent val,10 percent for test"
    
    
    def __init__(self, root, transforms1=None, transforms2=None, noise_level=0, mode='train', train_test_split = 0.8):
        # List the h5 files in root 
#         patient_id =list(range(0, 50))
        files = list(pathlib.Path(root).iterdir())
        files = sorted([str(i) for i in files if str(i).endswith(".h5")])[:] # indexing for patients with ktrans files
        imgs=[]
        if(noise_level == 0):
            self.xfms = transforms2
        else :
            self.xfms = transforms1

        if mode == "train":
            for filename in files[:int(train_test_split * 198)]:#len(files)-->200
                if filename[-3:] == '.h5':
                    imgs.append(filename)
            
                                    
        elif mode=='val':
            # for filename in files[int(train_test_split * len(files)):]:
            for filename in files[int(train_test_split * 198):int(train_test_split * 198) + int(0.1 * 198)]:
                if filename[-3:] == '.h5':
                    imgs.append(filename)
         
        elif mode=='test':
            # for filename in files[int(train_test_split * len(files)):]:
           for filename in files[int(train_test_split * 198) + int(0.1 * 198):int(train_test_split * 198) + int(0.2 * 198)]: 
                if filename[-3:] == '.h5':
                    imgs.append(filename)
            
        else:
            print("INVALID MODE")
                    
        # elif mode == 'test':
        #     for filename in files[int((train_test_split+0.2) * len(files)):]:
        #         if filename[-3:] == '.h5':
        #             imgs.append(filename)
                    
#                 if (os.path.isfile(os.path.join(root, str(i) + ".h5"))): 
#                     imgs.append(os.path.join(root, str(i) + ".h5"))

#         elif mode == "test":
#             for i in patient_id[int(0.8 * len(patient_id)):]:
#                 if (
#                     os.path.isfile(os.path.join(root, str(i) + ".h5"))):
#                     imgs.append(os.path.join(root, str(i) + ".h5"))

        self.examples = []


        for fname in imgs[:]:
            with h5py.File(fname,'r') as hf:
                fsvol = hf['t2']
                num_slices = fsvol.shape[-1]
                self.examples += [(fname, slice) for slice in range(3,13)]#
            

    def __len__(self):
        return len(self.examples)

    def normalize(self,arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    def mm(self,arr):
        print(np.min(arr),np.max(arr))
        return None
    def find_mid_threshold(self,image):
        min_val = np.min(image)
        max_val = np.max(image)
        mid_threshold = (min_val + max_val) / 2
        return mid_threshold +0.1

    
    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
        #print(fname,slice) 
        final_dic ={}
        with h5py.File(fname, 'r') as data:
            t2 = data['t2'][:,:,:].astype(np.float64)
            adc = data['ADC'][:,:,:].astype(np.float64)
            cz=data["CZ"][:,:,:].astype(np.float64)
            pz=data["PZ"][:,:,:].astype(np.float64)
            #pd = data['PD'][0,:,:,:].astype(np.float64)
    
            mask = data['Organ_mask'][:,:,:].astype(np.float64)
            lesion_mask =data['lesion_mask'][:,:,:].astype(np.float64)
            
            dce_01 = data['DCE1'][:,:,:].astype(np.float64)
           #dce_02 = data['DCE2'][0,:,:,:].astype(np.float64)
            #dce_03 = data['DCE3'][0,:,:,:].astype(np.float64)
            
            ### Extra time points
            dce10 = data['DCE10'][:,:,:].astype(np.float64)
            dce12 = data['DCE12'][:,:,:].astype(np.float64)
            dce15 = data['DCE15'][:,:,:].astype(np.float64)
            #dce17 = data['DCE_t17'][0,:,:,:].astype(np.float64)
            #dce20 = data['DCE_t20'][0,:,:,:].astype(np.float64)
            b400 = data['b400'][:,:,:].astype(np.float64)
            b800 = data['b800'][:,:,:].astype(np.float64)

        dic = {"T2": t2,"ADC":adc,'b400':b400,
                "DCE_01":dce_01,
                "DCE_t10":dce10,
                "DCE_t12":dce12,"DCE_t15":dce15,
               "organ_mask":mask,'lesion_mask':lesion_mask,'cz':cz,'pz':pz}#,"ktrans":ktrans}
        if self.xfms:
           
            trans_dic = self.xfms(dic)
            # print(trans_dic['T2'].shape)
            
            t2 = trans_dic['T2'][np.newaxis,:,:,slice].astype(np.float64)
            adc = trans_dic['ADC'][np.newaxis,:,:,slice].astype(np.float64)
            b400 = trans_dic['b400'][np.newaxis,:,:,slice].astype(np.float64)            
            mask = trans_dic['organ_mask'][np.newaxis,:,:,slice].astype(np.float64)
            cz = trans_dic['cz'][np.newaxis,:,:,slice].astype(np.float64)
            pz= trans_dic['pz'][np.newaxis,:,:,slice].astype(np.float64)
            lesion_mask = trans_dic['lesion_mask'][np.newaxis,:,:,slice].astype(np.float64)
            
            dce_01 = trans_dic['DCE_01'][np.newaxis,:,:,slice].astype(np.float64)
            dce_02 = trans_dic['DCE_02'][np.newaxis,:,:,slice].astype(np.float64)
            dce_03 = trans_dic['DCE_03'][np.newaxis,:,:,slice].astype(np.float64)
            ####
            dce10 = trans_dic['DCE_t10'][np.newaxis,:,:,slice].astype(np.float64)
            dce12 = trans_dic['DCE_t12'][np.newaxis,:,:,slice].astype(np.float64)
            dce15 = trans_dic['DCE_t15'][np.newaxis,:,:,slice].astype(np.float64)
            #dce17 = trans_dic['DCE_t17'][np.newaxis,:,:,slice].astype(np.float64)
            #dce20 = trans_dic['DCE_t20'][np.newaxis,:,:,slice].astype(np.float64)
            #b50 = trans_dic['b50'][np.newaxis,:,:,slice].astype(np.float64)
            # ktrans = trans_dic['ktrans'][np.newaxis,:,:,slice].astype(np.float64)
        else:
            trans_dic = dic
            t2 = trans_dic['T2'][np.newaxis,:,:,slice].astype(np.float64)
            adc = trans_dic['ADC'][np.newaxis,:,:,slice].astype(np.float64)
            b400 = trans_dic['b400'][np.newaxis,:,:,slice].astype(np.float64)
            
         #   pd = trans_dic['PD'][np.newaxis,:,:,slice].astype(np.float64)
            pz = trans_dic['pz'][np.newaxis,:,:,slice].astype(np.float64)
            mask = trans_dic['organ_mask'][np.newaxis,:,:,slice].astype(np.float64)
            lesion_mask = trans_dic['lesion_mask'][np.newaxis,:,:,slice].astype(np.float64)
            
            dce_01 = trans_dic['DCE_01'][np.newaxis,:,:,slice].astype(np.float64)
            #dce_02 = trans_dic['DCE_02'][np.newaxis,:,:,slice].astype(np.float64)
            #dce_03 = trans_dic['DCE_03'][np.newaxis,:,:,slice].astype(np.float64)
            ####
            dce10 = trans_dic['DCE_t10'][np.newaxis,:,:,slice].astype(np.float64)
            dce12 = trans_dic['DCE_t12'][np.newaxis,:,:,slice].astype(np.float64)
            dce15 = trans_dic['DCE_t15'][np.newaxis,:,:,slice].astype(np.float64)
            
        t2 = self.normalize(t2) 
        #pd = self.normalize(pd) 
        adc = self.normalize(adc) 
        b400 = self.normalize(b400) 
        
        dce_01 = self.normalize(dce_01) 
        dce10 = self.normalize(dce10) 
        dce12 = self.normalize(dce12)
        dce15 = self.normalize(dce15)

         ## Adaptive histogram equalization 
        limit = 0.01
        
        #t2 = exposure.equalize_adapthist(t2, clip_limit=limit)
        #adc = exposure.equalize_adapthist(adc, clip_limit=limit)
        #dce_01 = exposure.equalize_adapthist(dce_01, clip_limit=limit)
        #b400 = exposure.equalize_adapthist(b400, clip_limit=limit)
        # dce10 = exposure.equalize_adapthist(dce10, clip_limit=limit)
        # dce12 = exposure.equalize_adapthist(dce12, clip_limit=limit)
        # dce15 = exposure.equalize_adapthist(dce15, clip_limit=limit)

 
        # dce10=np.clip(dce10,0.3,1)
        # dce12=np.clip(dce12,0.3,1)
        # dce15=np.clip(dce15,0.3,1)
        
        t2 =  torch.from_numpy(t2)
        adc =  torch.from_numpy(adc)
        b400 =  torch.from_numpy(b400)
        
       # pd=  torch.from_numpy(pd)
        
        mask=torch.from_numpy(mask)
        lesion_mask = torch.from_numpy(lesion_mask)
        
        cz = torch.from_numpy(cz)
        pz = torch.from_numpy(pz)
        dce_01=  torch.from_numpy(dce_01)
        #dce_02 =  torch.from_numpy(dce_02)
        #dce_03 =  torch.from_numpy(dce_03)
        dce10 = torch.from_numpy(dce10)
        dce12 =  torch.from_numpy(dce12)
        dce15 = torch.from_numpy(dce15)

       
        return {"inp":torch.cat((t2,b400,adc,lesion_mask)),"dce_op":torch.cat((dce10,dce12,dce15))}
        # return t2,adc,b400,mask,lesion_mask,dce_01,dce10,dce12,dce15,pz,cz,str(fname.split('/')[-1]), slice
        
     