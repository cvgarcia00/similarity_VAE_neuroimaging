from .normalisation import normalization_min, normalization_exp, normalization_hist
from .load_database import transform_string
from config import load_config
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import yaml
import os
import glob
import pandas as pd
from collections import defaultdict
import random


config_file = os.environ.get('BETA_VAE_CONFIG', 'config.yaml')
config = load_config(config_file)

#LOAD HYPERPARAMETERS FROM CONFIG FILE
    
root_folder = config['loader']['load_folder']['dataset']
prefix = config['loader']['load_folder']['prefix']
extension = config['loader']['load_folder']['extension']
target_folder_name = config['loader']['load_folder']['target_folder_name']
splits = config['loader']['splits']
normalize = config['experiment']['normalization']
batch_size = config['model']['batch_size']
architecture = config['model']['architecture'] # Architecture of the model (VAE, CVAE)
path_ADNI_BIDS = config['loader']['load_ADNI_BIDS']




def find_pet(root_folder, prefix, extension, target_folder_name):
    """
    Script for finding _pet.nii files inside a main_folder. 

    ARGUMENTS:
    
        -root_folder: Path of the main folder that contains all the sub-folders
    
        -prefix: prefix of the file we want to find. Usually 'sub', 'wsub',
        'rsub', etc.
    
        -extension: Always set to '.nii' -NIfTI format-

        -targetFolderName: Name of folder where target files are contained.
        This is optional if we set Extension to be '_pet.nii' (for PET) or
        '_T1w.nii' (for MRI).

    """
    
    """
    This function is recursive. It enters the root folder and traverses
    the directory (one subject folder at a time). The first "if" condition checks
    if we have found the pet folder (which is found inside the sub-ADNI folders). If
    found, then it detects the target file (using the extension .nii and the prefix).
    If target_folder_name (usually "pet" or "anat") not found, then it moves to the next folder. 
    The function iterates through every single folder.
    """
    file_list = []
    
    #Get current name of the subject folder
    current_folder_name = os.path.basename(root_folder)
    
    #Check if current folder is target folder (PET or MRI)
    if current_folder_name == target_folder_name:
        
        #Pattern string of the files we look for (prefix-XXXX.extension)
        #The "*" indicates prefix"anything-in-between"extension
        pattern = os.path.join(root_folder, f"{prefix}*{extension}")
        #Find files with the patern prefix*extension, by checking every file
        #within the current folder
        files = glob.glob(pattern)
        #Add to list
        file_list.extend(files)
                
    #Get sub-folders inside current folder
    #It gets the path (f.path) of every f that is a folder if f is a directory and its name is not . or ..            
    sub_folders =  sorted([f.path for f in os.scandir(root_folder) if f.is_dir() and f.name not in {'.', '..'}])
    for sub_folder in sub_folders:
        #Call function for each subfolder to check if there are prefix*extension files inside
        #These subfolders should only be either "pet" or "anat"
        sub_folder_files = find_pet(sub_folder, prefix, extension, target_folder_name)
        file_list.extend(sub_folder_files)
    
    return file_list

def split_subjects_by_id(file_paths, splits, random_seed = 42):
    """
    This function helps us to avoid different scans from the same subject from
    falling into different sets.
    
    Split dataset by subject ID to avoid data leakage.
    
    Args:
        file_paths: List of file paths
        splits: List of split ratios [train, val, test]
        random_seed: Seed for reproducibility
        
    Returns:
        train_indices, val_indices, test_indices: List of indices for each split
    """
    
    # Extract subject IDs from file paths
    
    subject_to_indices = defaultdict(list)
    """
    Iterate throufh all file paths, saves the ID and the resulting index.
    
    If my files are:
    [0] /path/sub-ADNI002_S_0295/ses-bl/pet/wsub-ADNI002_S_0295_ses-bl_pet.nii
    [1] /path/sub-ADNI002_S_0295/ses-m06/pet/wsub-ADNI002_S_0295_ses-m06_pet.nii  
    [2] /path/sub-ADNI002_S_0413/ses-bl/pet/wsub-ADNI002_S_0413_ses-bl_pet.nii
    [3] /path/sub-ADNI002_S_0413/ses-m12/pet/wsub-ADNI002_S_0413_ses-m12_pet.nii
    [4] /path/sub-ADNI002_S_0295/ses-m12/pet/wsub-ADNI002_S_0295_ses-m12_pet.nii
    
    The resulting dictionary would be:
    
    subject_to_indices = {
        '002_S_0295': [0, 1, 4]
        '002_S_0413': [2, 3]
    }
    """
    for idx, file_path in enumerate(file_paths):
        try:
            id_ses = transform_string(file_path)
            subject_id = id_ses[0] # PTID (XXX_S_XXXX)
            subject_to_indices[subject_id].append(idx)
        except Exception as e:
            print(f"Warning: Could not extract subject ID from {file_path}: {e}")
            continue
        
    """
    This extracts only the keys from the previous dictionary, that is, each unique ID:
    
    unique_subjects = ['002_S_0295', '002_S_0413', ...]
    
    And then we shuffle this list.
    """
    unique_subjects = list(subject_to_indices.keys())
    
    # Set random seed for reproducibility
    
    random.seed(random_seed)
    random.shuffle(unique_subjects)
    
    # Calculate number of subjects for each split
    
    n_subjects = len(unique_subjects)
    n_train = int(splits[0] * n_subjects)
    n_val = int(splits[1] * n_subjects)
    n_test = n_subjects - n_train - n_val
    
    # Split subjects. First n_train subjects go to training,
    # the next n_val go to evaluation and the rest to test
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]
    
    print(f"Subject split summary:")
    print(f"Total subjects: {n_subjects}")
    print(f"Train subjects: {len(train_subjects)} ({len(train_subjects)/n_subjects:.2%})")
    print(f"Val subjects: {len(val_subjects)} ({len(val_subjects)/n_subjects:.2%})")
    print(f"Test subjects: {len(test_subjects)} ({len(test_subjects)/n_subjects:.2%})")
    
    # Get indices for each split
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    
    """
    For each subject in training, eval or test, we add ALL their images into
    train/eval/test_indices. If train_subjects = ['002_S_0295'] and this subject has indices
    [0, 1, 4] then, train_indcies will include [0, 1, 4].
    
    The final output would look like this:
    
    train_indices: [0, 1, 4, 7, 8, 12, ...]
    eval_indices: [2, 3, 9, 15, ...]
    test_indices: [5, 6, 10, 11, ...]
    
    where each index returns a unique path (corresponding to an scan) and avoids scans from
    the same subject from falling into diferent sets. A subject and ALL his visits go into the
    same set.
    """
    for subject in train_subjects:
        train_indices.extend(subject_to_indices[subject])
        
    for subject in val_subjects:
        val_indices.extend(subject_to_indices[subject])
    
    for subject in test_subjects:
        test_indices.extend(subject_to_indices[subject])
        
    print(f"Sample split summary:")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def _viscode_to_months(viscode):
    """This function transforms the VISCODE (bl, m03, etc.) to an integer that
    represents the months from baseline"""
    if pd.isna(viscode):
        return 0
    viscode = str(viscode).lower()
    if viscode == 'bl':
        return 0
    elif viscode.startswith('m'):
        #This extracts the number after m and transforms it to an integer
        try:
            return int(viscode[1:])
        except ValueError:
            return 0
    return 0
    
    

class BrainDataset(Dataset):
    def __init__(self, normalize = True):
        """
        Args: 
            - file_paths: list of file paths. I use find_pet function to obtain the
            paths of all files with certain names.
        """
        self.file_paths = find_pet(root_folder, prefix, extension, target_folder_name)
        self.normalize = normalize
        self.database = pd.read_csv(path_ADNI_BIDS)
        
        # Store initial count for reporting
        self.total_samples = len(self.file_paths)
        self.samples_with_complete_data = 0
        self.samples_with_missing_data = 0
        self.samples_with_adas13 = 0
        self.samples_without_adas13 = 0
        

        self.features = [
            score_name
            for config_key, score_name in config['results'].items()
            if config_key.startswith('feature_')
        ]
        
        #Add viscode to feature list but do not use it for analysis, only for processing

        """"
        Preprocess data for handling. Remove < and empty values. Transform
        to numerical data and set to NaN in case of issues.
        """
        for feature in self.features:
            self.database[feature] = (
                self.database[feature]
                .astype(str)  # Convert everything to string to handle
                .str.replace('<', '', regex=False)
                .str.replace('>', '', regex=False)
            )
            self.database[feature] = pd.to_numeric(self.database[feature], errors='coerce') # Coerce to NaN if not numeric
            
        """
        Add DX after preprocessing of data in order not to convert into NaN.
        Map CN to 0, MCI to 1 and Dementia to 2.
        """    
        self.features.append('DX')
        self.database['DX'] = self.database['DX'].map({'CN': 0, 'MCI': 1, 'Dementia': 2})
        
        """Add the months from baseline to the age"""
        
        #Compute months from baseline for each visit
        months_since_bl = self.database['VISCODE'].apply(_viscode_to_months)
        
        #Compute real age (age_bl + months/12) and update column
        self.database['AGE'] = self.database['AGE'] + (months_since_bl / 12)
            
        """
        Normalize volumetric features by the size of the brain of each subject
        to reduce noise variability. Use col loop to apply normalization to each feature.
        """
        colums_to_normalize = ["Ventricles", "Hippocampus", "Entorhinal", "Fusiform", "MidTemp"]
        normalization_column = "WholeBrain"
        
        for col in colums_to_normalize:
            self.database[col] = self.database[col] / self.database[normalization_column]
            
        """
        Apply Z-score normalization to ADAS13 values, so that average is 0 and standard deviation is 1.
        """
            
        self.database['ADAS13'] = (self.database['ADAS13'] - self.database['ADAS13'].mean() ) / self.database['ADAS13'].std() 
        self.database['CDRSB'] = (self.database['CDRSB'] - self.database['CDRSB'].mean() ) / self.database['CDRSB'].std()
        
        # Count samples with complete vs missing data
        self._count_data_completeness()
            
    def _count_data_completeness(self):
        """Count how many samples have complete data vs missing values"""
        all_features = self.features
        
        for file_path in self.file_paths:
            try:
                id_ses = transform_string(file_path)
                PTID = id_ses[0]
                VISCODE = id_ses[1]
                
                filter_mask = (self.database["PTID"] == PTID) & (self.database["VISCODE"] == VISCODE)
                matches = self.database.loc[filter_mask, all_features]
                
                if len(matches) > 0:
                    row = matches.iloc[0]
                    
                    # Check if all features are complete
                    if row.notna().all():
                        self.samples_with_complete_data += 1
                    else:
                        self.samples_with_missing_data += 1
                    
                    # Check specifically for ADAS13 (needed for correlation loss)
                    if pd.notna(row['ADAS13']):
                        self.samples_with_adas13 += 1
                    else:
                        self.samples_without_adas13 += 1
                else:
                    self.samples_with_missing_data += 1
                    self.samples_without_adas13 += 1
            except Exception:
                self.samples_with_missing_data += 1
                self.samples_without_adas13 += 1
    
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        """
        Load nifty file for subject [idx] and return it as a normalized tensor.
        """
        file_path = self.file_paths[idx] # Get path
        img = np.zeros((90, 110, 90)) # Create matrix with image dimensions
        img_file = nib.load(file_path) # Load the .nii file
        """Erase one row in X and Z dimensions. Add an extra empty row in Y dimensions. I do this to 
        change dimensions from (91,109,91) -> (90,110,90) which are easier to handle
        """
        img[:, :-1, :] = img_file.get_fdata()[1:, :, 1:] # 
        img[np.isnan(img)] = 0 # Replace NaN values with 0
        
        """
        Select a normalization pipeline.
        """
        if self.normalize:
            img = normalization_exp(img) # This is the one we use for experiments
            #img = normalization_min(img)
            #img = normalization_hist(img)
            
        """
        Force type to be float32 and transform into tensor to be handled by the model. I add an extra dimension 
        (1, 90, 110, 90) fo account for channels, which is expected by the VAE.
        """
        img = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
            
            
        """I extract the ID and Session from the path. Transforms from 'sub-ADNIXXX_SXXXX to
        XXX_S_XXXX and month visit from 'ses-MMXXX' to 'mXX'. If session is month 0, it is transformed
        to 'bl'.
        """
        id_ses = transform_string(file_path) # GET (ID, SES) FROM FILE PATH
        PTID = id_ses[0] #XXX_S_XXXX
        VISCODE = id_ses[1] #mXX
        
        
        """
        Here we create a boolean mask. We take the database and compare all PTIDs and VISCODEs to
        our current subject. We use .loc with the arguments "filter" (which selects the rows) and "self.features"
        (which selects our features). When we use .loc with filter, it takes the rows of the database corresponding to
        True (i.e., our subject). It returns 1 single column because there is only 1 entry of subject PTID and visit
        VISCODE. Squeeze transform from dataframe to pandas series. to_dict converts the series to a dictionary:
        {
            "ABETA": XX,
            "PTAU": YY,
            "ADAS13", XY,
            ...
        }    
        """
        filter = (self.database["PTID"] == PTID) & (self.database["VISCODE"] == VISCODE)
        features_dict = self.database.loc[filter, self.features].squeeze().to_dict()
        
        # We add ID and VISCODE to keep track of the subjects
        
        features_dict['PTID'] = PTID
        features_dict['VISCODE'] = VISCODE

        return img_tensor, features_dict

        

def get_dataloader(batch_size: int = 32, shuffle = True, num_workers = 16, random_seed = 42):

    """
    BrainDataset is like a box with all images and their data that we use for the model. The class (BrainDataset)
    is the set of instructions. Each call to the box is an instance (or object)
    """
    # Create the full dataset
    dataset = BrainDataset(normalize = True)
    
    # Print data completeness report
    print("\n" + "="*70)
    print("DATA COMPLETENESS REPORT")
    print("="*70)
    print(f"Total samples found (.nii files): {dataset.total_samples}")
    print(f"\nAll features complete: {dataset.samples_with_complete_data} ({dataset.samples_with_complete_data/dataset.total_samples*100:.2f}%)")
    print(f"At least one missing value: {dataset.samples_with_missing_data} ({dataset.samples_with_missing_data/dataset.total_samples*100:.2f}%)")
    print(f"\nADAS13 availability (required for correlation loss):")
    print(f"  - Samples with ADAS13: {dataset.samples_with_adas13} ({dataset.samples_with_adas13/dataset.total_samples*100:.2f}%)")
    print(f"  - Samples without ADAS13: {dataset.samples_without_adas13} ({dataset.samples_without_adas13/dataset.total_samples*100:.2f}%)")
    print(f"\nUsage in training:")
    print(f"  - Image reconstruction: ALL {dataset.total_samples} samples")
    print(f"  - Correlation loss (latent-ADAS13): Only {dataset.samples_with_adas13} samples with ADAS13")
    print("  - Samples with missing ADAS13 are filtered out in correlation calculation")
    print("="*70 + "\n")
    
    # Get subject-based splits in order to avoid data leakage
    
    train_indices, val_indices, test_indices = split_subjects_by_id(
        dataset.file_paths, splits, random_seed
    )
    
    # Create subset datasets
    train_set = Subset(dataset, train_indices)
    eval_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = True
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = True
    )
    
    return train_loader, eval_loader, test_loader

def verify_subject_separation(train_loader, eval_loader, test_loader):
    """
    Verify that no subject appears in multiple splits.
    """
    def get_subjects_from_loader(loader):
        subjects = set()
        for _, features_dict in loader:
            ptids = features_dict['PTID']
            if isinstance(ptids, list):
                subjects.update(ptids)
            else: subjects.add(ptids.item() if hasattr(ptids, 'item') else ptids)
        return subjects
    
    train_subjects = get_subjects_from_loader(train_loader)
    eval_subjects = get_subjects_from_loader(eval_loader)
    test_subjects = get_subjects_from_loader(test_loader)
    
    # Check for overlaps
    train_eval_overlap = train_subjects.intersection(eval_subjects)
    train_test_overlap = train_subjects.intersection(test_subjects)
    eval_test_overlap = eval_subjects.intersection(test_subjects)
    
    print(f"\nSubject separation verification:")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Eval subjects: {len(eval_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    
    if train_eval_overlap:
        print(f"WARNING: {len(train_eval_overlap)} subjects overlap between train and eval!")
    else:
        print("✓ No overlap between train and eval sets")
    
    if train_test_overlap:
        print(f"WARNING: {len(train_test_overlap)} subjects overlap between train and test!")
    else:
        print("✓ No overlap between train and test sets")
    
    if eval_test_overlap:
        print(f"WARNING: {len(eval_test_overlap)} subjects overlap between eval and test!")
    else:
        print("✓ No overlap between eval and test sets")
    
    return len(train_eval_overlap) == 0 and len(train_test_overlap) == 0 and len(eval_test_overlap) == 0         
    
    
    
    
    
    # train_size = int(splits[0] * len(dataset))
    # eval_size = int(splits[1] * len(dataset))
    # test_size = len(dataset) - train_size - eval_size
    
    # train_set, eval_set, test_set = random_split(dataset, [train_size, eval_size, test_size])
    
    # """
    # The DataLoader is what we use to load data in batches to train the model. It avoids loading everything at the
    # same time.
    # """
    
    # train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = True)
    # eval_loader = DataLoader(eval_set, batch_size = 48, shuffle = False, num_workers = num_workers, pin_memory = True) 
    # test_loader = DataLoader(test_set, batch_size = 48, shuffle = False, num_workers = num_workers, pin_memory = True) 
    
    # return train_loader, eval_loader, test_loader

