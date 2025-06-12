import torch.utils.data as data
from litsr import transforms
from litsr.data.image_folder import ImageFolder, RawImageFolder,Raw6xImageFolder
from litsr.utils.registry import DatasetRegistry
import torch as th
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from typing import Optional, Union
import numpy as np
import torch

@DatasetRegistry.register()
class SingleImageToPatchDataset(data.Dataset):
    def __init__(
        self,
        img_path,
        is_train,
        multi_spectral=False, channels=12,
        patch_size=None,
        rgb_range=1,
        repeat=1,
        data_length=None,
        cache=None,
        first_k=None,
        patch_num=2,
        mean=None,
        std=None,
        return_img_name=False,
        harmonization=False,
    ):
        assert not is_train ^ bool(
            patch_size
        ), "If is_train = True, the patch_size should be specified."

        self.is_train = is_train
        self.patch_size = patch_size
        self.dataset = ImageFolder(
            img_path,
            multi_spectral=multi_spectral, channels=channels,
            repeat=repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.patch_num = patch_num
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names = self.dataset.filenames
        self.harmonization = harmonization

    def get_patchs(self, hr):
        out = []
        hr = transforms.augment(hr)
        # extract two patches from each image
        for _ in range(self.patch_num):
            hr_patch = transforms.random_crop(hr, patch_size=self.patch_size)
            out.append(hr_patch)
        return out

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.is_train:
            patchs = self.get_patchs(img)
        else:
            patchs = [img]
        if self.harmonization:
            patchs = [self.harmonization_02(p) for p in patchs]
        #patchs = [self.harmonization_02(p) for p in patchs]
        patchs = [transforms.uint2single(p) for p in patchs]
        patchs = [transforms.single2tensor(p) * self.rgb_range for p in patchs]
        #print(np.max(patchs[0].cpu().numpy()))
        #print(np.min(patchs[0].cpu().numpy()))

        if self.mean and self.std:
            patchs = [
                transforms.normalize(p, self.mean, self.std, inplace=True)
                for p in patchs
            ]
        #print(np.max(patchs[0].cpu().numpy()))
        #print(np.min(patchs[0].cpu().numpy()))

        out = th.stack(patchs, 0)
        #plt.imsave('/nr/bamjo/user/msarmad/Work/SuperAI/BlindSRSNF/dump/image.png', out[0,:,:,:].cpu().numpy().transpose((1,2, 0))) #data_lr[k,:,:,:].cpu().numpy().transpose((1, 2, 0)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            return out, file_name
        else:
            return out

    def __len__(self):
        return len(self.dataset)




    def harmonization_02(self,
        image: np.ndarray,
        percentiles: Optional[float] = 50,
        seed: Optional[int] = 42,
        
    ):

        # Convert the image to 8-bit
        # transform channel axis to the first axis
        image = np.moveaxis(image, -1, 0)
        image = torch.Tensor(image).to(torch.uint8)

        # Set the random seed
        if seed is not None:
            torch.manual_seed(seed)

        # Set the log-normal model
        mean = np.array([0.33212655+0.02, 0.32608668+0.010, 0.35346831]) #+0.01 +0.005+0.01
        std = np.array(
            [
                [0.0059202, 0.00507745, 0.00445213],
                [0.00507745, 0.00494151, 0.00445722],
                [0.00445213, 0.00445722, 0.00485922],
                
            ]
        )
        model = stats.multivariate_normal(mean=mean, cov=std)


        def power_law(x, gamma, k=255):
            # Ensure gamma is a numpy array and has the same number of elements as there are channels in x
            gamma = np.array(gamma, dtype=np.float32).reshape(-1, 1, 1)  # Reshape gamma to (C, 1, 1)
            x = x / k  # Normalize x to range [0, 1]
            return np.power(x, 1 / gamma)   # Apply power law and scale back to original range

        # Make a sample of the model
        sample = model.rvs(size=100000)
        # model_value = np.percentile(sample, percentiles)
        
        model_value0 = np.percentile(sample[:,0], percentiles)
        model_value1 = np.percentile(sample[:,1], percentiles)
        model_value2 = np.percentile(sample[:,2], percentiles)
        model_value = [model_value0, model_value1, model_value2]
        image_out = power_law(image, model_value)
        #move the channel axis back to the last axis
        image_out = image_out.permute(1, 2, 0)
        # convert to array
        image_out = image_out.numpy()
        image_out = image_out * 255
        # convert to uint8
        image_out = image_out.astype(np.uint8)
        # Apply the power law to the image
        return  image_out#.to(device)


@DatasetRegistry.register()
class SingleRaw6ImageToPatchDataset(data.Dataset):
    def __init__(
        self,
        img_path,
        is_train,
        multi_spectral=False, channels=12,
        patch_size=None,
        rgb_range=1,
        repeat=1,
        data_length=None,
        cache=None,
        first_k=None,
        patch_num=2,
        mean=None,
        std=None,
        return_img_name=False,
        harmonization=False,
    ):
        assert not is_train ^ bool(
            patch_size
        ), "If is_train = True, the patch_size should be specified."

        self.is_train = is_train
        self.patch_size = patch_size
        self.dataset = Raw6xImageFolder(
            img_path,
            multi_spectral=multi_spectral, channels=channels,
            repeat=repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.patch_num = patch_num
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names = self.dataset.filenames
        self.harmonization = harmonization

    def get_patchs(self, hr):
        out = []
        hr = transforms.augment(hr)
        # extract two patches from each image
        for _ in range(self.patch_num):
            hr_patch = transforms.random_crop(hr, patch_size=self.patch_size)
            out.append(hr_patch)
        return out

    def __getitem__(self, idx):
        img = self.dataset[idx]
        img = img / 10000
        #hr_data = np.clip(hr_data, 0, 1)
        if self.is_train:
            patchs = self.get_patchs(img)
        else:
            patchs = [img]
        if self.harmonization:
            pass
        #patchs = [self.harmonization_02(p) for p in patchs]
        #patchs = [transforms.uint2single(p) for p in patchs]
        patchs = [transforms.single2tensor(p) * self.rgb_range for p in patchs]
        #print(np.max(patchs[0].cpu().numpy()))
        #print(np.min(patchs[0].cpu().numpy()))

        #hr_data = hr_data /15000
        # Run the model 
        # clip the hr_data from 0 to 1
        #hr_data = np.clip(hr_data, 0, 1)
        if self.mean and self.std:
            patchs = [
                transforms.normalize(p, self.mean, self.std, inplace=True)
                for p in patchs
            ]
        #print(np.max(patchs[0].cpu().numpy()))
        #print(np.min(patchs[0].cpu().numpy()))

        out = th.stack(patchs, 0)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            return out, file_name
        else:
            return out

    def __len__(self):
        return len(self.dataset)



@DatasetRegistry.register()
class SingleRawImageToPatchDataset(data.Dataset):
    def __init__(
        self,
        img_path,
        is_train,
        multi_spectral=False, channels=12,
        patch_size=None,
        rgb_range=1,
        repeat=1,
        data_length=None,
        cache=None,
        first_k=None,
        patch_num=2,
        mean=None,
        std=None,
        return_img_name=False,
        harmonization=False,
    ):
        assert not is_train ^ bool(
            patch_size
        ), "If is_train = True, the patch_size should be specified."

        self.is_train = is_train
        self.patch_size = patch_size
        self.dataset = RawImageFolder(
            img_path,
            multi_spectral=multi_spectral, channels=channels,
            repeat=repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.patch_num = patch_num
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names = self.dataset.filenames
        self.harmonization = harmonization

    def get_patchs(self, hr):
        out = []
        hr = transforms.augment(hr)
        # extract two patches from each image
        for _ in range(self.patch_num):
            hr_patch = transforms.random_crop(hr, patch_size=self.patch_size)
            out.append(hr_patch)
        return out

    def __getitem__(self, idx):
        img = self.dataset[idx]
        img = img / 10000
        #hr_data = np.clip(hr_data, 0, 1)
        if self.is_train:
            patchs = self.get_patchs(img)
        else:
            patchs = [img]
        if self.harmonization:
            pass
        #patchs = [self.harmonization_02(p) for p in patchs]
        #patchs = [transforms.uint2single(p) for p in patchs]
        patchs = [transforms.single2tensor(p) * self.rgb_range for p in patchs]
        #print(np.max(patchs[0].cpu().numpy()))
        #print(np.min(patchs[0].cpu().numpy()))

        #hr_data = hr_data /15000
        # Run the model 
        # clip the hr_data from 0 to 1
        #hr_data = np.clip(hr_data, 0, 1)
        if self.mean and self.std:
            patchs = [
                transforms.normalize(p, self.mean, self.std, inplace=True)
                for p in patchs
            ]
        #print(np.max(patchs[0].cpu().numpy()))
        #print(np.min(patchs[0].cpu().numpy()))

        out = th.stack(patchs, 0)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            return out, file_name
        else:
            return out

    def __len__(self):
        return len(self.dataset)



@DatasetRegistry.register()
class DoubleRawImageToPatchDataset(data.Dataset):
    def __init__(
        self,
        img_path,
        is_train,
        multi_spectral=False, channels=12,
        patch_size=None,
        rgb_range=1,
        repeat=1,
        data_length=None,
        cache=None,
        first_k=None,
        patch_num=2,
        mean=None,
        std=None,
        return_img_name=False,
        harmonization=False,
    ):
        assert not is_train ^ bool(
            patch_size
        ), "If is_train = True, the patch_size should be specified."

        self.is_train = is_train
        self.patch_size = patch_size
        self.dataset1 = ImageFolder(
            img_path[0],
            multi_spectral=multi_spectral, channels=channels,
            repeat=repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )

        self.dataset2 = RawImageFolder(
            img_path[1],
            multi_spectral=multi_spectral, channels=channels,
            repeat=repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.patch_num = patch_num
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names1 = self.dataset1.filenames
        self.file_names2 = self.dataset2.filenames
        self.harmonization = harmonization

    def get_patchs(self, hr):
        out = []
        hr = transforms.augment(hr)
        # extract two patches from each image
        for _ in range(self.patch_num):
            hr_patch = transforms.random_crop(hr, patch_size=self.patch_size)
            out.append(hr_patch)
        return out
    def harmonization_02(self,
            image: np.ndarray,
            percentiles: Optional[float] = 50,
            seed: Optional[int] = 42,
            
        ):

            # Convert the image to 8-bit
            # transform channel axis to the first axis
            image = np.moveaxis(image, -1, 0)
            image = torch.Tensor(image).to(torch.uint8)

            # Set the random seed
            if seed is not None:
                torch.manual_seed(seed)

            # Set the log-normal model
            mean = np.array([0.33212655+0.02, 0.32608668+0.010, 0.35346831]) #+0.01 +0.005+0.01
            std = np.array(
                [
                    [0.0059202, 0.00507745, 0.00445213],
                    [0.00507745, 0.00494151, 0.00445722],
                    [0.00445213, 0.00445722, 0.00485922],
                    
                ]
            )
            model = stats.multivariate_normal(mean=mean, cov=std)


            def power_law(x, gamma, k=255):
                # Ensure gamma is a numpy array and has the same number of elements as there are channels in x
                gamma = np.array(gamma, dtype=np.float32).reshape(-1, 1, 1)  # Reshape gamma to (C, 1, 1)
                x = x / k  # Normalize x to range [0, 1]
                return np.power(x, 1 / gamma)   # Apply power law and scale back to original range

            # Make a sample of the model
            sample = model.rvs(size=100000)
            # model_value = np.percentile(sample, percentiles)
            
            model_value0 = np.percentile(sample[:,0], percentiles)
            model_value1 = np.percentile(sample[:,1], percentiles)
            model_value2 = np.percentile(sample[:,2], percentiles)
            model_value = [model_value0, model_value1, model_value2]
            image_out = power_law(image, model_value)
            #move the channel axis back to the last axis
            image_out = image_out.permute(1, 2, 0)
            # convert to array
            image_out = image_out.numpy()
            image_out = image_out * 255
            # convert to uint8
            image_out = image_out.astype(np.uint8)
            # Apply the power law to the image
            return  image_out#.to(device)
    def __getitem__(self, idx):
        # Select the image from both datasets randomly
        if np.random.rand() > 0.5:
            img = self.dataset1[idx]
            dataset = 1
        else:
            img = self.dataset2[idx]
            img = img / 15000
            dataset = 2


        #hr_data = np.clip(hr_data, 0, 1)
        if self.is_train:
            patchs = self.get_patchs(img)
        else:
            patchs = [img]
        if dataset == 1:
            if self.harmonization:
                patchs = [self.harmonization_02(p) for p in patchs]
            patchs = [transforms.uint2single(p) for p in patchs]
        

        patchs = [transforms.single2tensor(p) * self.rgb_range for p in patchs]


        #hr_data = hr_data /15000
        # Run the model 
        # clip the hr_data from 0 to 1
        #hr_data = np.clip(hr_data, 0, 1)
        if self.mean and self.std:
            patchs = [
                transforms.normalize(p, self.mean, self.std, inplace=True)
                for p in patchs
            ]
        #print(np.max(patchs[0].cpu().numpy()))
        #print(np.min(patchs[0].cpu().numpy()))

        out = th.stack(patchs, 0)

        if self.return_img_name:
            if dataset == 1:
                file_name = self.file_names1[idx % (len(self.dataset1) // self.repeat)]
            else:
                file_name = self.file_names2[idx % (len(self.dataset2) // self.repeat)]
            return out, file_name
        else:
            return out

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)
