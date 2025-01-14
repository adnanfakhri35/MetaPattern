def parse_folder_as_list(folder_path):
    """
    Parse a folder path and create a data list similar to what parse_data_list would return
    """
    import os
    from collections import OrderedDict
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder path {folder_path} does not exist")
        
    # Valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Get all valid image files
    files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(valid_extensions):
            files.append(os.path.join(folder_path, f))
    
    if not files:
        raise ValueError(f"No valid image files found in {folder_path}")
    
    # Create OrderedDict for compatibility with original code
    data_file_list = OrderedDict()
    face_labels = OrderedDict()
    
    for idx, f in enumerate(sorted(files)):
        data_file_list[idx] = f
        # Determine label based on folder name
        face_labels[idx] = 1 if 'spoof' in folder_path.lower() else 0
    
    data_file_list.size = len(files)
    face_labels.size = len(files)
    
    return data_file_list, face_labels

def get_dataset_from_list(data_list_path, dataset_cls, train=True, transform=None, num_frames=1, root_dir=''):
    """
    Create a dataset from either a list file or a folder path
    """
    import os
    import logging
    from data.data_loader import parse_data_list
    
    try:
        # Check if the path is a directory
        if os.path.isdir(os.path.join(root_dir, data_list_path)):
            data_file_list, face_labels = parse_folder_as_list(os.path.join(root_dir, data_list_path))
            logging.info(f"Parsed folder {data_list_path}: found {data_file_list.size} images")
        else:
            # Assume it's a CSV file
            data_file_list, face_labels = parse_data_list(data_list_path)
            logging.info(f"Parsed list file {data_list_path}: found {data_file_list.size} entries")
    except Exception as e:
        logging.error(f"Error parsing data source {data_list_path}: {str(e)}")
        raise ValueError(f"No valid datasets could be created from {data_list_path}")

    if not data_file_list or data_file_list.size == 0:
        raise ValueError(f"No data found in {data_list_path}")
    
    dataset_list = []
    
    for i in range(data_file_list.size):
        try:
            face_label = int(face_labels.get(i) == 0)  # 0 means real face
            file_path = data_file_list.get(i)
            
            # If the path is already absolute (from folder parsing), don't join with root_dir
            if os.path.isabs(file_path):
                full_path = file_path
            else:
                full_path = os.path.join(root_dir, file_path)
            
            if not os.path.exists(full_path):
                logging.warning(f"Skip {full_path} (not exists)")
                continue
                
            dataset = dataset_cls(full_path, face_label, transform, num_frames=num_frames)
            
            if len(dataset) == 0:
                logging.warning(f"Skip {full_path} (zero elements)")
                continue
                
            dataset_list.append(dataset)
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not dataset_list:
        raise ValueError(f"No valid datasets could be created from {data_list_path}")
    
    final_dataset = torch.utils.data.ConcatDataset(dataset_list)
    logging.info(f"Created dataset with {len(final_dataset)} total samples from {len(dataset_list)} directories")
    
    return final_dataset
