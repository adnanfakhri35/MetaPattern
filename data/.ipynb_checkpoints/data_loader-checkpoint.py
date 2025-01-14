import os
import sys
import logging
import torch
import pandas as pd
from pathlib import Path

def parse_data_list(data_list_path):
    """
    Parse image data from OULU-NPU dataset structure with additional validation and detailed logging
    """
    data_file_list = []
    face_labels = []
    invalid_files = []
    
    logging.info(f"Parsing data from: {data_list_path}")
    logging.info(f"Directory exists: {os.path.exists(data_list_path)}")
    
    if not os.path.exists(data_list_path):
        raise FileNotFoundError(f"Data path does not exist: {data_list_path}")

    try:
        # Handle real images directory
        real_dir = os.path.join(data_list_path, 'real')
        logging.info(f"Checking real directory: {real_dir}")
        logging.info(f"Real directory exists: {os.path.exists(real_dir)}")
        
        if os.path.exists(real_dir):
            all_files = os.listdir(real_dir)
            logging.info(f"Total files in real directory: {len(all_files)}")
            real_images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            logging.info(f"Valid image files in real directory: {len(real_images)}")
            
            for img_file in real_images:
                img_path = os.path.join('real', img_file)
                full_path = os.path.join(data_list_path, img_path)
                if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                    data_file_list.append(img_path)
                    face_labels.append(0)
                else:
                    invalid_files.append(full_path)
                    logging.warning(f"Invalid or empty file: {full_path}")

        # Handle spoof images directory
        spoof_dir = os.path.join(data_list_path, 'spoof')
        logging.info(f"Checking spoof directory: {spoof_dir}")
        logging.info(f"Spoof directory exists: {os.path.exists(spoof_dir)}")
        
        if os.path.exists(spoof_dir):
            all_files = os.listdir(spoof_dir)
            logging.info(f"Total files in spoof directory: {len(all_files)}")
            spoof_images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            logging.info(f"Valid image files in spoof directory: {len(spoof_images)}")
            
            for img_file in spoof_images:
                img_path = os.path.join('spoof', img_file)
                full_path = os.path.join(data_list_path, img_path)
                if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                    data_file_list.append(img_path)
                    face_labels.append(1)
                else:
                    invalid_files.append(full_path)
                    logging.warning(f"Invalid or empty file: {full_path}")

        total_images = len(data_file_list)
        logging.info(f"Total valid images found: {total_images}")
        logging.info(f"Real images: {face_labels.count(0)}")
        logging.info(f"Spoof images: {face_labels.count(1)}")
        
        if total_images == 0:
            logging.error("No valid images found. Directory structure:")
            for root, dirs, files in os.walk(data_list_path):
                logging.error(f"Directory: {root}")
                logging.error(f"  Subdirectories: {dirs}")
                logging.error(f"  Files: {len(files)} files")
            raise ValueError(
                f"No valid images found in {data_list_path}. "
                f"Please ensure the directory contains valid images in 'real' and/or 'spoof' subdirectories."
            )
            
    except Exception as e:
        logging.error(f"Error parsing directory {data_list_path}: {str(e)}")
        raise
        
    return data_file_list, face_labels

def get_dataset_from_list(data_list_path, dataset_cls, transform, num_frames=100, root_dir=''):
    """
    Create dataset from OULU-NPU directory structure with enhanced validation
    """
    logging.info(f"Creating dataset from: {data_list_path}")
    logging.info(f"Using root directory: {root_dir or data_list_path}")
    
    # If root_dir is empty, use data_list_path as root
    if not root_dir:
        root_dir = data_list_path
    
    # Ensure paths are absolute
    root_dir = os.path.abspath(root_dir)
    data_list_path = os.path.abspath(data_list_path)
    
    data_file_list, face_labels = parse_data_list(data_list_path)
    
    try:
        # Create dataset with validation
        dataset = dataset_cls(root_dir, data_file_list, transform=transform, num_frames=num_frames)
        
        # Validate first few items in dataset
        if len(dataset) > 0:
            try:
                first_item = dataset[0]
                logging.info(f"Successfully loaded first item from dataset")
                logging.info(f"First item shape: {first_item[0].shape if isinstance(first_item[0], torch.Tensor) else 'unknown'}")
            except Exception as e:
                logging.error(f"Error loading first item from dataset: {str(e)}")
                raise ValueError(f"Dataset creation succeeded but items cannot be loaded: {str(e)}")
        
        logging.info(f"Successfully created dataset with {len(dataset)} samples")
        
        # Verify dataset is not empty
        if len(dataset) == 0:
            raise ValueError(
                "Dataset is empty after creation. Possible issues:\n"
                f"1. No valid images found in {data_list_path}\n"
                f"2. Dataset class ({dataset_cls.__name__}) filter criteria too strict\n"
                f"3. num_frames ({num_frames}) value incompatible with available data"
            )
            
        return dataset
        
    except Exception as e:
        logging.error(f"Error creating dataset: {str(e)}")
        logging.error(f"Dataset class: {dataset_cls.__name__}")
        logging.error(f"Root dir exists: {os.path.exists(root_dir)}")
        logging.error(f"Number of file paths: {len(data_file_list)}")
        raise

def verify_paths(config):
    """Verify all paths in config exist"""
    paths_to_check = [
        ('DATA.ROOT_DIR', config.DATA.ROOT_DIR),
        ('DATA.TRAIN', config.DATA.TRAIN) if hasattr(config.DATA, 'TRAIN') else None,
        ('DATA.TEST', config.DATA.TEST) if hasattr(config.DATA, 'TEST') else None,
        ('DATA.VAL', config.DATA.VAL) if hasattr(config.DATA, 'VAL') else None
    ]
    
    for path_name, path in paths_to_check:
        if path is not None:
            if not os.path.exists(path):
                logging.error(f"Path {path_name} does not exist: {path}")
                raise FileNotFoundError(f"Required path {path_name} not found: {path}")
            else:
                logging.info(f"Verified path {path_name}: {path}")

def get_data_loader(config):
    """Get data loader with enhanced validation"""
    # Verify all paths first
    verify_paths(config)
    
    batch_size = config.DATA.BATCH_SIZE
    num_workers = 0 if config.DEBUG else config.DATA.NUM_WORKERS
    dataset_cls = zip_dataset.__dict__[config.DATA.DATASET]
    dataset_root_dir = config.DATA.ROOT_DIR
    dataset_subdir = config.DATA.SUB_DIR
    face_dataset_dir = os.path.join(dataset_root_dir, dataset_subdir)
    num_frames_train = config.TRAIN.NUM_FRAMES
    num_frames_test = config.TEST.NUM_FRAMES

    assert config.DATA.TRAIN or config.DATA.TEST, "Please provide at least a data_list"

    test_data_transform = VisualTransform(config)
    
    if config.DATA.TEST:
        test_dataset = get_dataset_from_list(
            config.DATA.TEST, 
            dataset_cls,
            test_data_transform,
            num_frames_test,
            root_dir=face_dataset_dir
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
        return test_loader
        
    else:
        assert config.DATA.TRAIN, "CONFIG.DATA.TRAIN should be provided"
        
        aug_transform = get_augmentation_transforms(config)
        train_data_transform = VisualTransform(config, aug_transform)

        train_dataset = get_dataset_from_list(
            config.DATA.TRAIN,
            dataset_cls, 
            train_data_transform,
            num_frames_train,
            root_dir=face_dataset_dir
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size,
            num_workers=num_workers, 
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        assert config.DATA.VAL, "CONFIG.DATA.VAL should be provided"
        
        val_dataset = get_dataset_from_list(
            config.DATA.VAL,
            dataset_cls,
            test_data_transform, 
            num_frames=num_frames_test,
            root_dir=face_dataset_dir
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )

        return train_loader, val_loader