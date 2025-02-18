import torch
import h5py
import numpy as np
import pathlib
import torchvision.transforms as transforms
class SliceData(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transforms=None, mode='train', train_test_split=0.8):
        print("ROOT",root)
        files = list(pathlib.Path(root).iterdir())
        
        files = sorted([str(i) for i in files])
        # print("files",files)
        imgs = []
        self.xfms = transforms
        if mode == "train":
            for filename in files[:int(train_test_split * len(files))]:
                if filename[-3:] == '.h5':
                    imgs.append(filename)
        elif mode == 'test' or 'validation':
            for filename in files[int(train_test_split * len(files)):][:]:
                if filename[-3:] == '.h5':
                    imgs.append(filename)

        self.examples = []
        for fname in imgs:
            with h5py.File(fname, 'r') as hf:
                fsvol = hf['T2']
                num_slices = fsvol.shape[-1]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        # print("len",len(self.examples))
        return len(self.examples)

    def __getitem__(self, i):
        print("i",i)
        fname, slice = self.examples[i] 
        print("filename",fname,slice)

        with h5py.File(fname, 'r') as data:
            t2 = torch.from_numpy(data['T2'][0, :, :, slice].astype(np.float64))
            adc = torch.from_numpy(data['ADC'][0, :, :, slice].astype(np.float64))
            pd = torch.from_numpy(data['PD'][0, :, :, slice].astype(np.float64))
            dce_01 = torch.from_numpy(data['DCE_01'][0, :, :, slice].astype(np.float64))
            dce_02 = torch.from_numpy(data['DCE_02'][0,:,:,slice].astype(np.float64))
            dce_03 = torch.from_numpy(data['DCE_03'][0,:,:,slice].astype(np.float64))
            
         # Resize images to 128x128
        resize = transforms.Resize((128, 128))
        t2 = resize(t2.unsqueeze(0))
        adc = resize(adc.unsqueeze(0))
        pd = resize(pd.unsqueeze(0))
        dce_01 = resize(dce_01.unsqueeze(0))
        dce_02 = resize(dce_02.unsqueeze(0))
        dce_03 = resize(dce_03.unsqueeze(0))

        # print(t2.shape)
        # Remove the channel dimension after resizing
        t2 = t2.squeeze(0)
        adc = adc.squeeze(0)
        pd = pd.squeeze(0)
        dce_01 = dce_01.squeeze(0)
        dce_02 = dce_02.squeeze(0)
        dce_03 = dce_03.squeeze(0)
        # print(t2.shape)
        # Crop the center 60x60 region from each image
        def crop_to_center(image):
            cx, cy = image.shape[0] // 2, image.shape[1] // 2
            return image[cx-30:cx+30, cy-30:cy+30]

        t2_cropped = crop_to_center(t2)
        adc_cropped = crop_to_center(adc)
        pd_cropped = crop_to_center(pd)
        dce_01_cropped = crop_to_center(dce_01)

        # Create a 160x160 image filled with zeros
        def create_zeros_image():
            return torch.zeros(128, 128)

        # Create empty 160x160 images for each modality
        t2_160 = create_zeros_image()
        adc_160 = create_zeros_image()
        pd_160 = create_zeros_image()
        dce_01_160 = create_zeros_image()

        # # Place the cropped 60x60 image in the center of the 160x160 image
        # def place_in_center(zeros_image, cropped_image):
        #     cx, cy = zeros_image.shape[0] // 2, zeros_image.shape[1] // 2
        #     zeros_image[50:110, 50:110] = cropped_image  # Place the 60x60 crop in the center
        #     return zeros_image

        # Place the cropped 60x60 image in the center of the 128x128 image
        def place_in_center(zeros_image, cropped_image):
            start_x = (zeros_image.shape[0] - cropped_image.shape[0]) // 2
            start_y = (zeros_image.shape[1] - cropped_image.shape[1]) // 2
            end_x = start_x + cropped_image.shape[0]
            end_y = start_y + cropped_image.shape[1]
            zeros_image[start_x:end_x, start_y:end_y] = cropped_image  # Place the crop in the center
            return zeros_image


        t2_160 = place_in_center(t2_160, t2_cropped)
        adc_160 = place_in_center(adc_160, adc_cropped)
        pd_160 = place_in_center(pd_160, pd_cropped)
        dce_01_160 = place_in_center(dce_01_160, dce_01_cropped)

        # Return the data in a dictionary as before
        # print(t2.shape,t2_160.shape)
        # data_lst = {
        #     'A': torch.cat((t2[None,:,:],pd[None,:,:],adc[None,:,:],dce_01[None,:,:],t2_160[None, :, :], pd_160[None, :, :], adc_160[None, :, :], dce_01_160[None, :, :]), axis=0),
        #     'B': torch.cat((dce_02[None,:,:],dce_03[None,:,:]),axis=0),
        #     'DX': i  # You can also return the index or any additional info as needed
        # }

        batch = torch.cat((t2[None,:,:],pd[None,:,:],adc[None,:,:],dce_01[None,:,:],t2_160[None, :, :], pd_160[None, :, :], adc_160[None, :, :], dce_01_160[None, :, :],dce_02[None,:,:],dce_03[None,:,:]),axis=0)
        print(fname,slice)
        return batch,{"y":np.array((1))}
        
