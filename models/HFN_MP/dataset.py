import logging
import os
import zipfile

import cv2
import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from data.data_loader import parse_data_list
from data.transforms import VisualTransform
from data.transforms import get_augmentation_transforms
from data.zip_dataset import ZipDataset as _ZipDataset
import functools


class ZipDataset_(torch.utils.data.Dataset):
    def __init__(self, zip_file_path, face_label, transforms=None, num_frames=1, train=True):
        self.zip_file_path = zip_file_path
        self.decode_flag = cv2.IMREAD_UNCHANGED
        self.train = train
        self.face_label = face_label

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms


        self.image_list_in_zip = []
        with zipfile.ZipFile(self.zip_file_path, "r") as zip:
            lst = zip.namelist()
            exts = ['png', 'jpg']
            for ext in exts:
                self.image_list_in_zip += list(filter(lambda x: x.lower().endswith(ext), lst))

        if len(self.image_list_in_zip) > 0:
            self.image_list_in_zip = self.image_list_in_zip[:num_frames]
        self.len = len(self.image_list_in_zip)



    def __read_image_from_zip__(self, index):
        image_name_in_zip = self.image_list_in_zip[index]
        with zipfile.ZipFile(self.zip_file_path, "r") as zip:
            bytes_ = zip.read(image_name_in_zip)
            bytes_ = np.frombuffer(bytes_, dtype=np.uint8)
            im = cv2.imdecode(bytes_, self.decode_flag)  # cv2 image
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (256,256))
            im =  Image.fromarray(im)
            return im# F.to_tensor(im)



    def __getitem__(self, index):

        im = self.__read_image_from_zip__(index)
        tensor = self.transforms(im)
        # print(tensor.shape)
        tensor = tensor.to(torch.float)
        target = {
            'face_label': self.face_label,
            'depth': self.face_label*torch.ones([1,32,32,], dtype=torch.float32)
        }
        return index, tensor, target, self.zip_file_path

    def __len__(self):
        return self.len



class ZipDataset(_ZipDataset):
    def __init__(self, zip_file_path, face_label, transform=None, num_frames=1, train=True):
        _ZipDataset.__init__(self, zip_file_path, face_label, transform=transform, num_frames=num_frames)

    def __getitem__(self, index):
        index, tensor, target, self.zip_file_path = _ZipDataset.__getitem__(self, index)
        target['depth'] =  self.face_label * torch.ones([1, 32, 32, ], dtype=torch.float32)
        return index, tensor, target, self.zip_file_path

class ZipDatasetPixelMC(ZipDataset):
    """
        ZipDataset with Pixel-wise target (Multi channel)
    """

    def __init__(self, zip_file_path, face_label, transform, num_frames=1000, config=None):
        super(ZipDatasetPixelMC, self).__init__(zip_file_path, face_label, transform, num_frames)

        self.config = config
    def __getitem__(self, index):

        im = self.__read_image_from_zip__(index)
        pixel_maps_size = 32

        pixel_maps_size = [32, 16, 8, 4, 2]
        pixel_maps = []
        for s in pixel_maps_size:
            pixel_maps.append(self.face_label * torch.ones([s, s]))

        channels = channel_list(im, self.config)
        channels = [ self.transform(im) for im in channels ]

        channels = torch.cat(channels, 0)
        target = {
            'face_label': self.face_label,
            'pixel_maps': pixel_maps
        }
        return index, channels, target, self.zip_file_path


def channel_list(bgr_im, config=None):
    channel_list = []
    if config.MODEL.CHANNELS.RGB:
        channel_list.append(bgr_im)
    if config.MODEL.CHANNELS.HSV:
        img_hsv = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2HSV)
        channel_list.append(img_hsv)

    if config.MODEL.CHANNELS.YCRCB:
        img_ycrcb = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2YCrCb)

        channel_list.append(img_ycrcb)

    if config.MODEL.CHANNELS.LAB:
        lab_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2LAB)
        channel_list.append(lab_image)

    if config.MODEL.CHANNELS.YUV:
        yuv_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2YUV)
        channel_list.append(yuv_image)

    if config.MODEL.CHANNELS.XYZ:
        xyz_image = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2XYZ)
        channel_list.append(xyz_image)

    return channel_list

'''
Additional code to change the ZipDataset Function 
'''
class FolderDatasetPixelMC(Dataset):
    """
    Dataset that loads images from a folder with Pixel-wise target (Multi channel)
    """
    def __init__(self, folder_path, face_label, transform, num_frames=1000, config=None):
        self.folder_path = folder_path
        self.face_label = face_label
        self.transform = transform
        self.config = config
        self.image_paths = []
        
        # Valid image extensions
        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        # Populate the image paths list
        # If folder_path is a directory, scan for images
        if os.path.isdir(folder_path):
            self._load_image_paths_from_dir(folder_path)
        # If folder_path is a file, add it directly if it's a valid image
        elif os.path.isfile(folder_path) and folder_path.lower().endswith(self.valid_extensions):
            self.image_paths.append(folder_path)
        else:
            raise ValueError(f"Invalid path: {folder_path}")
        
        # Limit the number of frames if specified
        if num_frames and num_frames < len(self.image_paths):
            self.image_paths = self.image_paths[:num_frames]
    
    def _load_image_paths_from_dir(self, directory):
        """Load image paths from a directory"""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(self.valid_extensions):
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        try:
            # Read and process image
            image_path = self.image_paths[index]
            im = self._read_image(image_path)
            
            # Convert to numpy array for channel processing
            im_np = np.array(im)
            
            # Process channels
            channels = self._process_channels(im_np)
            channels = torch.cat(channels, 0)
            
            # Generate pixel maps
            pixel_maps = [self.face_label * torch.ones([s, s]) for s in [32, 16, 8, 4, 2]]
            
            target = {
                'face_label': self.face_label,
                'pixel_maps': pixel_maps
            }
            
            return index, channels, target, image_path
            
        except Exception as e:
            logging.error(f"Error processing image {self.image_paths[index]}: {str(e)}")
            raise
    
    def _read_image(self, image_path):
        """Read and preprocess image"""
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = img.resize((256, 256))  # Ensure consistent size
            return img
    
    def _process_channels(self, im_np):
        import torchvision.transforms as transforms
        """Process different color channels"""
        channels = []
        
        if isinstance(im_np, PIL.Image.Image):
            im_np = np.array(im_np)

        # Apply the transform to each channel
        return [transforms.ToTensor()(ch) for ch in im_np.transpose(2, 0, 1)]
    
#         # Convert to different color spaces based on config
#         if hasattr(self.config.MODEL.CHANNELS, 'RGB') and self.config.MODEL.CHANNELS.RGB:
#             channels.append(Image.fromarray(im_np))
        
#         if hasattr(self.config.MODEL.CHANNELS, 'HSV') and self.config.MODEL.CHANNELS.HSV:
#             hsv = cv2.cvtColor(im_np, cv2.COLOR_RGB2HSV)
#             channels.append(Image.fromarray(hsv))
        
#         if hasattr(self.config.MODEL.CHANNELS, 'LAB') and self.config.MODEL.CHANNELS.LAB:
#             lab = cv2.cvtColor(im_np, cv2.COLOR_RGB2LAB)
#             channels.append(Image.fromarray(lab))
        
#         if hasattr(self.config.MODEL.CHANNELS, 'YUV') and self.config.MODEL.CHANNELS.YUV:
#             yuv = cv2.cvtColor(im_np, cv2.COLOR_RGB2YUV)
#             channels.append(Image.fromarray(yuv))
            
#         if not channels:  # If no channels specified, default to RGB
#             channels.append(Image.fromarray(im_np))
            
#         # Apply transforms to each channel
#         return [self.transform(ch) for ch in channels]

# def get_dataset_from_folder(folder_path, dataset_class, transform=None, root_dir='', num_frames=1):
#     dataset = dataset_class(transform=transform, root_dir=root_dir, num_frames=num_frames)
#     if isinstance(folder_path, list):
#         for path in folder_path:
#             full_path = os.path.join(root_dir, path)
#             label = 1 if 'fake' in path.lower() else 0
#             dataset.add_data_from_folder(full_path, label)
#     else:
#         full_path = os.path.join(root_dir, folder_path)
#         label = 1 if 'fake' in folder_path.lower() else 0
#         dataset.add_data_from_folder(full_path, label)
#     return dataset

def get_dataset_from_list(data_list_path, dataset_cls, train=True, transform=None, num_frames=1, root_dir=''):
    """
    Create a dataset from a list of file paths
    
    Args:
        data_list_path: path to data list file or directory
        dataset_cls: dataset class to use
        train: whether this is for training
        transform: transforms to apply to images
        num_frames: number of frames to use
        root_dir: root directory for data
    
    Returns:
        ConcatDataset containing all individual datasets
    """
    
    logging.info(f"Creating dataset from: {data_list_path}")
    
    if not os.path.exists(data_list_path):
        raise FileNotFoundError(f"Path does not exist: {data_list_path}")
    
    data_files, labels = parse_data_list(data_list_path)
    if not data_files:
        raise ValueError(f"No valid data found in {data_list_path}")
    
    dataset_list = []
    for file_path, label in zip(data_files, labels):
        try:
            full_path = os.path.join(root_dir, file_path) if root_dir else file_path
            dataset = dataset_cls(full_path, label, transform, num_frames=num_frames)
            if len(dataset) > 0:
                dataset_list.append(dataset)
                logging.info(f"Added dataset from {full_path}")
        except Exception as e:
            logging.warning(f"Skipping {file_path}: {str(e)}")
            continue
    
    if not dataset_list:
        raise ValueError(f"No valid datasets could be created from {data_list_path}")
    
    return torch.utils.data.ConcatDataset(dataset_list)
    
#     dataset_list = []
    
#     # Iterate through all files
#     for i, file_path in enumerate(data_file_list):
#         try:
#             # Determine the label (0 for real face, 1 for spoof)
#             face_label = 0 if face_labels[i] == 0 else 1
            
#             # Construct full path
#             full_path = os.path.join(root_dir, file_path)
#             logging.info(f"Processing file {i+1}/{len(data_file_list)}: {full_path}")

            
#             # Check if path exists
#             if not os.path.exists(full_path):
#                 logging.warning(f"Skipping {full_path} (does not exist)")
#                 continue
            
#             # Create dataset for this path
#             dataset = dataset_cls(full_path, face_label, transform, num_frames=num_frames)
            
#             # Check if dataset is empty
#             if len(dataset) == 0:
#                 logging.warning(f"Skipping {full_path} (zero elements)")
#                 continue
            
#             dataset_list.append(dataset)
#             logging.info(f"Successfully added dataset with {len(dataset)} elements")
            
#         except Exception as e:
#             logging.error(f"Error processing {file_path}: {str(e)}")
#             continue
            
#     if not dataset_list:
#         logging.error("No valid datasets were created. Check the following:")
#         logging.error(f"1. Data list path exists: {os.path.exists(data_list_path)}")
#         logging.error(f"2. Root directory exists: {os.path.exists(root_dir)}")
#         logging.error(f"3. Data list format is correct")
#         logging.error(f"4. Image files exist in the specified locations")
#         raise ValueError(f"No valid datasets could be created from {data_list_path}")
    
    # Combine all datasets
#     final_dataset = torch.utils.data.ConcatDataset(dataset_list)
#     logging.info(f"Created dataset with {len(final_dataset)} total samples from {len(dataset_list)} directories")
    
#     return final_dataset

'''
the modified code ends here
'''

# def get_dataset_from_list(data_list_path, dataset_cls, train=True, transform=None, num_frames=1, root_dir='', ):
#     # TODO: hard code it right now
#     data_file_list, face_labels = parse_data_list(data_list_path)

#     num_file = data_file_list.size
#     dataset_list = []

#     for i in range(num_file):
#         face_label = int(face_labels.get(i)==0) # 0 means real face and non-zero represents spoof
#         file_path = data_file_list.get(i)
#         # zip_path = os.path.join(file_path, root_path)
#         zip_path = root_dir + file_path
#         if not os.path.exists(zip_path):
#             logging.warning("Skip {} (not exists)".format(zip_path))
#             continue
#         else:
#             dataset = dataset_cls(zip_path, face_label, transform, num_frames=num_frames)
#             if len(dataset) == 0:
#                 logging.warning("Skip {} (zero elements)".format(zip_path))
#                 continue
#             else:
#                 dataset_list.append(dataset)
#     final_dataset = torch.utils.data.ConcatDataset(dataset_list)
#     return final_dataset


def get_data_loader(config):


    aug_transform = get_augmentation_transforms(config)
    train_data_transform = VisualTransform(config, aug_transform)
    test_data_transform = VisualTransform(config)
    root_dir = os.path.join(config.DATA.ROOT_DIR, config.DATA.SUB_DIR)
    Dataset = functools.partial(FolderDatasetPixelMC, config=config)

    batch_size = config.DATA.BATCH_SIZE
    if config.DATA.TEST:
        tgt_test_data = get_dataset_from_list(config.DATA.TEST, Dataset, transform=test_data_transform,
                                              root_dir=root_dir, num_frames=config.TEST.NUM_FRAMES)


        tgt_dataloader = DataLoader(tgt_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        return tgt_dataloader
    src1_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_1
    src1_train_data_real_list = config.DATA.TRAIN_SRC_REAL_1

#     src2_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_2
#     src2_train_data_real_list = config.DATA.TRAIN_SRC_REAL_2

#     src3_train_data_fake_list = config.DATA.TRAIN_SRC_FAKE_3
#     src3_train_data_real_list = config.DATA.TRAIN_SRC_REAL_3

    num_frames = config.TRAIN.NUM_FRAMES
    src1_train_data_fake = get_dataset_from_list(src1_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    src1_train_data_real = get_dataset_from_list(src1_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    # src2_train_data_fake = get_dataset_from_list(src2_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    # src2_train_data_real = get_dataset_from_list(src2_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    # src3_train_data_fake = get_dataset_from_list(src3_train_data_fake_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)
    # src3_train_data_real = get_dataset_from_list(src3_train_data_real_list, Dataset, transform=train_data_transform, num_frames=config.TRAIN.NUM_FRAMES, root_dir=root_dir)



    tgt_test_data_list = config.DATA.TARGET_DATA
    tgt_test_data = get_dataset_from_list(tgt_test_data_list, Dataset, transform=test_data_transform, root_dir=root_dir, num_frames = config.TEST.NUM_FRAMES)


    print('Load Target Data')

    src1_train_dataloader_fake = DataLoader(src1_train_data_fake,
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(src1_train_data_real,
                                            batch_size=batch_size, shuffle=True)
    # src2_train_dataloader_fake = DataLoader(src2_train_data_fake,
    #                                         batch_size=batch_size, shuffle=True)
    # src2_train_dataloader_real = DataLoader(src2_train_data_real,
    #                                         batch_size=batch_size, shuffle=True)
    # src3_train_dataloader_fake = DataLoader(src3_train_data_fake,
    #                                         batch_size=batch_size, shuffle=True)
    # src3_train_dataloader_real = DataLoader(src3_train_data_real,
    #                                         batch_size=batch_size, shuffle=True)
    tgt_dataloader = DataLoader(tgt_test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
            tgt_dataloader
           # src2_train_dataloader_fake, src2_train_dataloader_real, \
           # src3_train_dataloader_fake, src3_train_dataloader_real, \