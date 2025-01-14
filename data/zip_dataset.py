import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_list, transform, num_frames=1):
        """
        Args:
            root_dir (str): Root directory containing the image files
            data_list (list): List of image paths relative to root_dir
            transform: Image transformation pipeline
            num_frames (int): Number of frames to sample (default 1 for image data)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.decode_flag = cv2.IMREAD_UNCHANGED
        self.data_list = data_list
        
        # Extract labels from paths (0 for real, 1 for spoof)
        self.labels = [1 if 'spoof' in path else 0 for path in data_list]
        
        # Sample frames if needed
        if len(self.data_list) > num_frames:
            sample_indices = np.linspace(0, len(self.data_list)-1, num=num_frames, dtype=int)
            self.data_list = [self.data_list[i] for i in sample_indices]
            self.labels = [self.labels[i] for i in sample_indices]
        
        self.len = len(self.data_list)
        
        # Verify initialization
        print(f"ZipDataset initialized with:")
        print(f"- Root directory: {self.root_dir}")
        print(f"- Number of images: {self.len}")
        print(f"- Sample path: {self.data_list[0] if self.len > 0 else 'No images'}")

    def __read_image__(self, index):
        image_path = os.path.join(self.root_dir, self.data_list[index])
        im = cv2.imread(image_path, self.decode_flag)
        if im is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return im

    def __getitem__(self, index):
        im = self.__read_image__(index)  # cv2 image, format [H, W, C], BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        tensor = self.transform(im)
        tensor = tensor.to(torch.float)
        target = {
            'face_label': self.labels[index]
        }
        return index, tensor, target, self.data_list[index]

    def __len__(self):
        return self.len

class ZipDatasetPixelFPN(ZipDataset):
    """
    Dataset with Pixel target for FPN
    """
    def __init__(self, root_dir, data_list, transform, num_frames=1):
        super(ZipDatasetPixelFPN, self).__init__(root_dir, data_list, transform, num_frames)

    def __getitem__(self, index):
        im = self.__read_image__(index)
        # No RGB to BGR
        pixel_maps_size = [32, 16, 8, 4, 2]
        pixel_maps = []
        for s in pixel_maps_size:
            pixel_maps.append(self.labels[index] * torch.ones([s,s]))
        im = self.transform(im)
        target = {
            'face_label': self.labels[index],
            'pixel_maps': pixel_maps
        }
        return index, im, target, self.data_list[index]

class ZipDatasetPixel(ZipDataset):
    """
    Dataset with Pixel-wise target
    """
    def __init__(self, root_dir, data_list, transform, num_frames=1):
        super(ZipDatasetPixel, self).__init__(root_dir, data_list, transform, num_frames)

    def __getitem__(self, index):
        im = self.__read_image__(index)
        pixel_maps_size = 32
        pixel_maps = self.labels[index] * torch.ones([pixel_maps_size,pixel_maps_size])
        im = self.transform(im)
        target = {
            'face_label': self.labels[index],
            'pixel_maps': pixel_maps
        }
        return index, im, target, self.data_list[index]

class ZipDatasetMultiChannel(ZipDataset):
    """
    Dataset for RGB+HSV or RGB+YUV
    """
    def __init__(self, root_dir, data_list, transform, num_frames=1):
        super(ZipDatasetMultiChannel, self).__init__(root_dir, data_list, transform, num_frames)

    def __getitem__(self, index):
        im = self.__read_image__(index)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im = self.transform(im)
        target = {
            'face_label': self.labels[index]
        }
        return index, im, target, self.data_list[index]