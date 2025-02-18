import os
# os.chdir('/media/Data16T/Kishore/RESVIT/Resvit_for_add_tp/')
# print(os.getcwd())
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

#### Class for loader
class SliceData(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    
    "Loader hsa been changed for 70 percent train,20 percent val,10 percent for test
    """
    
    def __init__(self, root, transforms1=None, transforms2=None, noise_level=0, mode='train', train_test_split = 0.7,norm=True):
        files = list(pathlib.Path(root).iterdir())
        # print(files)
        files = sorted([str(i) for i in files if str(i).endswith(".h5") and os.path.basename(i) != '0052.h5' and os.path.basename(i) != '0080.h5' and os.path.basename(i) != '0138.h5' and os.path.basename(i) != '0056.h5' and os.path.basename(i) != '0025.h5'])[:] # indexing for training patients with ktrans

        # files = sorted([str(i) for i in files if str(i).endswith(".h5") and os.path.basename(i) != '0025.h5' ])[:]
        
        self.norm = norm
        imgs=[]
        if(noise_level == 0):
            self.xfms = transforms2
        else :
            self.xfms = transforms1

        if mode == "train":
            for filename in files[:int(train_test_split * len(files))]:
                # print("train",filename)
                if str(filename[-3:]) == '.h5':
                    imgs.append(filename)
                    
        elif mode=='val' :#or mode =='test':
            print("into val")
            # for filename in files[int(train_test_split * len(files)):]:
            for filename in files[int(train_test_split * len(files)):int((train_test_split+0.15) * len(files))]:
                # print(filename)
                if str(filename[-3:]) == '.h5':
                    # print("its h5")
                    imgs.append(filename)        
        else:
            print("into test")
            for filename in files[int((train_test_split+0.15) * len(files)):]:
                # print(filename)
                if str(filename[-3:]) == '.h5':
                    # print("its h5")
                    imgs.append(filename)
        self.examples = []


        for fname in sorted(imgs[0:]):
            with h5py.File(fname,'r') as hf:
                # print(fname,hf.keys())
                fsvol = hf['T2']
                num_slices = fsvol.shape[-1]
                self.examples += [(fname, slice) for slice in range(num_slices)]
            

    def __len__(self):
        # print("len",len(self.examples))
        
        return len(self.examples)

    def normalize(self,arr,which_norm ='min_max'):
        if which_norm =='min_max':
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        elif which_norm =='z_score':
            return (arr - np.mean(arr)) / np.std(arr)
        else:
            print("Specify within minmax or z_score norm")
            
    # def z_norm(self,arr):
    #     return (arr - np.mean(arr)) / np.std(arr)

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
        # print("fname,slice",fname.split('/')[-1],slice,i)
        
        final_dic ={}
        with h5py.File(fname, 'r') as data:
            if not 'DCE_t25' in data.keys():
                
                print(fname,data.keys())   
            t2 = data['T2'][0,:,:,:].astype(np.float64)
            adc = data['ADC'][0,:,:,:].astype(np.float64)
            pd = data['PD'][0,:,:,:].astype(np.float64)
            dce_01 = data['DCE1'][0,:,:,:].astype(np.float64)
            dce_02 = data['DCE2'][0,:,:,:].astype(np.float64)
            dce_03 = data['DCE3'][0,:,:,:].astype(np.float64)

            #### DWI
            dwi_50 = data['DWI_50'][0,:,:,:].astype(np.float64)
            dwi_400 = data['DWI_400'][0,:,:,:].astype(np.float64)
            dwi_800 = data['DWI_800'][0,:,:,:].astype(np.float64)
            ### Extra time points
            dce10 = data['DCE_t10'][0,:,:,:].astype(np.float64)
            dce12 = data['DCE_t12'][0,:,:,:].astype(np.float64)
            dce15 = data['DCE_t15'][0,:,:,:].astype(np.float64)
            dce17 = data['DCE_t17'][0,:,:,:].astype(np.float64)
            dce20 = data['DCE_t20'][0,:,:,:].astype(np.float64)

            dce25 = data['DCE_t25'][0,:,:,:].astype(np.float64)
            dce30 = data['DCE_t30'][0,:,:,:].astype(np.float64)
            dce35 = data['DCE_t35'][0,:,:,:].astype(np.float64)
            dce40 = data['DCE_t40'][0,:,:,:].astype(np.float64)
            dce42 = data['DCE_t42'][0,:,:,:].astype(np.float64)
            
            
            # ktrans = data['ktrans'][0,:,:,:].astype(np.float64)
            
            # optional
            # thresh_ktrans  =data['thresh_5_ktrans'][0,:,:,:].astype(np.float64)

            
        # dic = {"T2": t2,"ADC":adc,
        #         "PD": pd, "DCE_01":dce_01,
        #         "DCE_02": dce_02, "DCE_03":dce_03,"DCE_t10":dce10,
        #         "DCE_t12":dce12,"DCE_t15":dce15,"DCE_t17":dce17,"DCE_t20":dce20,"ktrans":ktrans,"thresh_ktrans":thresh_ktrans}

        dic = {"T2": t2,"ADC":adc,"PD": pd, 
               "DWI_50":dwi_50,"DWI_400":dwi_400,"DWI_800":dwi_800,
               "DCE_01":dce_01,"DCE_02": dce_02, "DCE_03":dce_03,
               "DCE_t10":dce10,"DCE_t12":dce12,"DCE_t15":dce15,"DCE_t17":dce17,"DCE_t20":dce20,
               "DCE_t25":dce25,"DCE_t30":dce30,"DCE_t35":dce35,"DCE_t40":dce40,"DCE_t42":dce42,}
               # "ktrans":ktrans}
        # ,"thresh_ktrans":thresh_ktrans}
        
        if self.xfms:
            trans_dic = self.xfms(dic)
            # print(trans_dic['T2'].shape)
            
            t2 = trans_dic['T2'][np.newaxis,:,:,slice].astype(np.float64)
            adc = trans_dic['ADC'][np.newaxis,:,:,slice].astype(np.float64)
            pd = trans_dic['PD'][np.newaxis,:,:,slice].astype(np.float64)
            #### dwi 
            dwi_50 = trans_dic['DWI_50'][np.newaxis,:,:,slice].astype(np.float64)
            dwi_400 = trans_dic['DWI_400'][np.newaxis,:,:,slice].astype(np.float64)
            dwi_800 = trans_dic['DWI_800'][np.newaxis,:,:,slice].astype(np.float64)
            ######
            dce_01 = trans_dic['DCE_01'][np.newaxis,:,:,slice].astype(np.float64)
            dce_02 = trans_dic['DCE_02'][np.newaxis,:,:,slice].astype(np.float64)
            dce_03 = trans_dic['DCE_03'][np.newaxis,:,:,slice].astype(np.float64)
            #### Early time points
            dce10 = trans_dic['DCE_t10'][np.newaxis,:,:,slice].astype(np.float64)
            dce12 = trans_dic['DCE_t12'][np.newaxis,:,:,slice].astype(np.float64)
            dce15 = trans_dic['DCE_t15'][np.newaxis,:,:,slice].astype(np.float64)
            dce17 = trans_dic['DCE_t17'][np.newaxis,:,:,slice].astype(np.float64)
            dce20 = trans_dic['DCE_t20'][np.newaxis,:,:,slice].astype(np.float64)

            #### Late time points
            dce25 = trans_dic['DCE_t25'][np.newaxis,:,:,slice].astype(np.float64)
            dce30 = trans_dic['DCE_t30'][np.newaxis,:,:,slice].astype(np.float64)
            dce35 = trans_dic['DCE_t35'][np.newaxis,:,:,slice].astype(np.float64)
            dce40 = trans_dic['DCE_t40'][np.newaxis,:,:,slice].astype(np.float64)
            dce42 = trans_dic['DCE_t42'][np.newaxis,:,:,slice].astype(np.float64)
            #### ktrans 
            
            # ktrans = trans_dic['ktrans'][np.newaxis,:,:,slice].astype(np.float64)

        else:
            trans_dic = dic
            t2 = trans_dic['T2'][np.newaxis,:,:,slice].astype(np.float64)
            adc = trans_dic['ADC'][np.newaxis,:,:,slice].astype(np.float64)
            pd = trans_dic['PD'][np.newaxis,:,:,slice].astype(np.float64)
            #### dwi 
            dwi_50 = trans_dic['DWI_50'][np.newaxis,:,:,slice].astype(np.float64)
            dwi_400 = trans_dic['DWI_400'][np.newaxis,:,:,slice].astype(np.float64)
            dwi_800 = trans_dic['DWI_800'][np.newaxis,:,:,slice].astype(np.float64)
            ####
            dce_01 = trans_dic['DCE_01'][np.newaxis,:,:,slice].astype(np.float64)
            dce_02 = trans_dic['DCE_02'][np.newaxis,:,:,slice].astype(np.float64)
            dce_03 = trans_dic['DCE_03'][np.newaxis,:,:,slice].astype(np.float64)

            #### Early time points 
            dce10 = trans_dic['DCE_t10'][np.newaxis,:,:,slice].astype(np.float64)
            dce12 = trans_dic['DCE_t12'][np.newaxis,:,:,slice].astype(np.float64)
            dce15 = trans_dic['DCE_t15'][np.newaxis,:,:,slice].astype(np.float64)
            dce17 = trans_dic['DCE_t17'][np.newaxis,:,:,slice].astype(np.float64)
            dce20 = trans_dic['DCE_t20'][np.newaxis,:,:,slice].astype(np.float64)

            #### Late time points
            dce25 = trans_dic['DCE_t25'][np.newaxis,:,:,slice].astype(np.float64)
            dce30 = trans_dic['DCE_t30'][np.newaxis,:,:,slice].astype(np.float64)
            dce35 = trans_dic['DCE_t35'][np.newaxis,:,:,slice].astype(np.float64)
            dce40 = trans_dic['DCE_t40'][np.newaxis,:,:,slice].astype(np.float64)
            dce42 = trans_dic['DCE_t42'][np.newaxis,:,:,slice].astype(np.float64)

            ### ktrans 
            # ktrans = trans_dic['ktrans'][np.newaxis,:,:,slice].astype(np.float64)
            # thresh_ktrans = trans_dic['thresh_ktrans'][np.newaxis,:,:,slice].astype(np.float64)

        sub_dce_02 = np.abs(dce_02 - dce_01)
        sub_dce_03 = np.abs(dce_03 - dce_01)

        norm_sub_dce_02 = self.normalize(sub_dce_02,which_norm = 'min_max')
        norm_sub_dce_03 = self.normalize(sub_dce_03,which_norm = 'min_max')
        
        # thres_2 = self.find_mid_threshold(norm_sub_dce_02)
        # thres_3 = self.find_mid_threshold(norm_sub_dce_03)
        
        # thres_sub_dce2 = torch.from_numpy(norm_sub_dce_02 > thres_2)
        # thres_sub_dce3 = torch.from_numpy(norm_sub_dce_03 > thres_3)
        # # print("before",t2.dtype)
        if self.norm:
            
            t2 =  torch.from_numpy(self.normalize(t2,which_norm = 'min_max'))
            adc =  torch.from_numpy(self.normalize(adc,which_norm = 'min_max'))
            pd=  torch.from_numpy(self.normalize(pd,which_norm = 'min_max'))
            #### dwi
            dwi_50 =  torch.from_numpy(self.normalize(dwi_50,which_norm = 'min_max'))
            dwi_400=  torch.from_numpy(self.normalize(dwi_400,which_norm = 'min_max'))
            dwi_800 =  torch.from_numpy(self.normalize(dwi_800,which_norm = 'min_max'))
            ####
            dce_01=  torch.from_numpy(self.normalize(dce_01,which_norm = 'min_max'))
            dce_02 =  torch.from_numpy(self.normalize(dce_02,which_norm = 'min_max'))
            dce_03 =  torch.from_numpy(self.normalize(dce_03,which_norm = 'min_max'))
            ## early tp 
            dce10 = torch.from_numpy(self.normalize(dce10,which_norm = 'min_max'))
            dce12 =  torch.from_numpy(self.normalize(dce12,which_norm = 'min_max'))
            dce15 = torch.from_numpy(self.normalize(dce15,which_norm = 'min_max'))
            dce17 = torch.from_numpy(self.normalize(dce17,which_norm = 'min_max'))
            dce20 = torch.from_numpy(self.normalize(dce20,which_norm = 'min_max'))

            ### late tp
            dce25 = torch.from_numpy(self.normalize(dce25,which_norm = 'min_max'))
            dce30 =  torch.from_numpy(self.normalize(dce30,which_norm = 'min_max'))
            dce35 = torch.from_numpy(self.normalize(dce35,which_norm = 'min_max'))
            dce40 = torch.from_numpy(self.normalize(dce40,which_norm = 'min_max'))
            dce42 = torch.from_numpy(self.normalize(dce42,which_norm = 'min_max'))
            ## ktrans
            
            # ktrans = torch.from_numpy(self.z_norm(ktrans))
            
            # thresh_ktrans = torch.from_numpy(self.z_norm(thresh_ktrans))
            # thresh_ktrans = torch.from_numpy(thresh_ktrans) # NOTE: dONOT NORMALIZE SEG MAPS
            
        else:
            t2 =  torch.from_numpy(t2)
            adc =  torch.from_numpy(adc)
            pd=  torch.from_numpy(pd)
            ### dwi 
            dwi_50 =  torch.from_numpy(dwi_50)
            dwi_400=  torch.from_numpy(dwi_400)
            dwi_800 =  torch.from_numpy(dwi_800)
            ### dce1,2,3
            dce_01=  torch.from_numpy(dce_01)
            dce_02 =  torch.from_numpy(dce_02)
            dce_03 =  torch.from_numpy(dce_03)
            ### early tp 
            dce10 = torch.from_numpy(dce10)
            dce12 =  torch.from_numpy(dce12)
            dce15 = torch.from_numpy(dce15)
            dce17 = torch.from_numpy(dce17)
            dce20 = torch.from_numpy(dce20)

            ### late tp
            dce25 = torch.from_numpy(dce25)
            dce30 =  torch.from_numpy(dce30)
            dce35 = torch.from_numpy(dce35)
            dce40 = torch.from_numpy(dce40)
            dce42 = torch.from_numpy(dce42)
            
            # ktrans = torch.from_numpy(self.normalize(ktrans))
            # thresh_ktrans = torch.from_numpy(self.normalize(thresh_ktrans))
            # thresh_ktrans = torch.from_numpy(thresh_ktrans) # NOTE: dONOT NORMALIZE SEG MAPS
        
        sub_dce_02 = torch.from_numpy(sub_dce_02)
        sub_dce_03 = torch.from_numpy(sub_dce_03)
         
        norm_sub_dce_02 = torch.from_numpy(norm_sub_dce_02)
        norm_sub_dce_03 = torch.from_numpy(norm_sub_dce_03)
        # print("thresh_ktrans shape",thresh_ktrans.shape)
        
        # print("after",t2.dtype)
        # print(dce10.dtype)
        # resvit_many = {'A':torch.cat((t2,pd,adc,dce_01),axis=0),'B':torch.cat((sub_dce_02,sub_dce_03),axis=0),'C':ktrans}
        # resvit_many = {'A':torch.cat((t2,pd,adc,dce_01),axis=0),'B':torch.cat((dce_02,dce_03),axis=0),'C':ktrans} # normal one 
        
        #  for multi op prediction
        # resvit_many = {'A':torch.cat((t2,pd,adc,dce_01),axis=0),'B':torch.cat((dce10,dce12,dce15,dce17,dce20),axis=0)}
        # ktrans as channel one 
        
        resvit_many = {'A':torch.cat((t2,pd,adc,dce_01),axis=0),
                       'B':torch.cat((dce10,dce12,dce15,dce17,dce20),axis=0)} # 4 channels
        
        resvit_many_late= {'A':torch.cat((t2,pd,adc,dce_01),axis=0),
                           'B':torch.cat((dce10,dce12,dce15,dce17,dce20,dce25,dce30,dce35,dce40,dce42),axis=0)}

        structural = {'A':t2,'B':torch.cat((dce10,dce12,dce15,dce17,dce20),axis=0)}
        # structural_t2_t1 = {'A':torch.cat((t2,dce_01),axis=0),'B':torch.cat((dce10,dce12,dce15,dce17,dce20),axis=0)}

        non_structural = {'A':torch.cat((t2,dwi_50),axis=0),'B':torch.cat((dce10,dce12,dce15,dce17,dce20),axis=0)}
                           # 'C':ktrans} # 4 channels
        
        batch = torch.cat((t2,pd,adc,dce_01,dce10,dce12,dce15,dce17,dce20))
        # inp_to_ktrans = {'A':torch.cat((t2,pd,adc,dce_01),axis=0),'B':ktrans}
        
        # dce_to_ktrans  ={'A':torch.cat((dce10,dce12,dce15,dce17,dce20),axis=0),'B':ktrans}
        
        # resvit_many = {'A':torch.cat((t2,pd,adc,dce_01,sub_dce_02,sub_dce_03),axis=0),'B':torch.cat((dce10,dce12,dce15,dce17,dce20),axis=0),'C':ktrans} # 6 channels
        
        # resvit_many = {'A':torch.cat((t2,pd,adc,dce_01,sub_dce_02,sub_dce_03),axis=0),'B':torch.cat((ktrans,dce10,dce12,dce15,dce17,dce20),axis=0)}
        # resvit_many = {'A':torch.cat((t2,pd,adc,dce_01,sub_dce_02,sub_dce_03),axis=0),'B':torch.cat((dce10,dce12,dce15,dce17,dce20),axis=0)}

        # resvit_one = {'A':torch.cat((t2,pd,adc,dce_01,thres_sub_dce2,thres_sub_dce3),axis=0),'B':torch.cat((norm_sub_dce_02,norm_sub_dce_03),axis=0),'C':torch.cat((dce_02,dce_03))} # residual one
        
#         dic = {"T2": torch.from_numpy(t2),"ADC":torch.from_numpy(adc),
#                 "PD": torch.from_numpy(pd), "DCE_01":torch.from_numpy(dce_01),
#                 "DCE_02": torch.from_numpy(dce_02), "DCE_03":torch.from_numpy(dce_03)}
        
        return batch,{"y":np.array((1))}#resvit_many#non_structural#structural#resvit_many_late#resvit_one#dic