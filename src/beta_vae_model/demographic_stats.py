"""
Script to extract demographic statistics (age and sex) for train, validation, and test sets.
This creates a table for paper publication with summary statistics.
"""

import pandas as pd
import numpy as np
import yaml
from collections import defaultdict
import os
import glob
import random
from config import load_config

# Load configuration
config_file = os.environ.get('BETA_VAE_CONFIG', 'config.yaml')
config = load_config(config_file)

root_folder = config['loader']['load_folder']['dataset']
prefix = config['loader']['load_folder']['prefix']
extension = config['loader']['load_folder']['extension']
target_folder_name = config['loader']['load_folder']['target_folder_name']
splits = config['loader']['splits']
path_participants = os.path.join(root_folder, 'participants.tsv')


def transform_string(file_path):
    """Extract subject ID and session from file path."""
    file_name = os.path.basename(file_path)
    # Format: rsub-ADNI072S4103_ses-M000_trc-18FFDG_rec-coregavg_pet.nii
    # Extract ADNI ID: 072S4103
    if 'ADNI' in file_name:
        # Find the ADNI ID
        start_idx = file_name.find('ADNI') + 4  # Skip 'ADNI'
        # Extract until first underscore or space after ADNI
        end_idx = file_name.find('_', start_idx)
        if end_idx == -1:
            end_idx = file_name.find(' ', start_idx)
        if end_idx == -1:
            end_idx = len(file_name)
        
        adni_id = file_name[start_idx:end_idx]  # e.g., '072S4103'
        
        # Format to match participants.tsv format: XXX_S_XXXX
        # From '072S4103' to '072_S_4103'
        if 'S' in adni_id:
            parts = adni_id.split('S')
            if len(parts) == 2:
                subject_id = f"{parts[0]}_S_{parts[1]}"
            else:
                subject_id = adni_id
        else:
            subject_id = adni_id
        
        # Extract session
        session = None
        if 'ses-' in file_name:
            session_start = file_name.find('ses-') + 4
            session_end = file_name.find('_', session_start)
            if session_end == -1:
                session_end = file_name.find('.', session_start)
            session = file_name[session_start:session_end]
        
        return [subject_id, session]
    return [None, None]


def find_pet(root_folder, prefix, extension, target_folder_name):
    """Find PET files in directory structure."""
    file_list = []
    current_folder_name = os.path.basename(root_folder)
    
    if current_folder_name == target_folder_name:
        pattern = os.path.join(root_folder, f"{prefix}*{extension}")
        files = glob.glob(pattern)
        file_list.extend(files)
    
    sub_folders = sorted([f.path for f in os.scandir(root_folder) if f.is_dir() and f.name not in {'.', '..'}])
    for sub_folder in sub_folders:
        sub_folder_files = find_pet(sub_folder, prefix, extension, target_folder_name)
        file_list.extend(sub_folder_files)
    
    return file_list


def split_subjects_by_id(file_paths, splits, random_seed=42):
    """Split dataset by subject ID to avoid data leakage."""
    subject_to_indices = defaultdict(list)
    
    for idx, file_path in enumerate(file_paths):
        try:
            id_ses = transform_string(file_path)
            subject_id = id_ses[0]
            if subject_id:
                subject_to_indices[subject_id].append(idx)
        except Exception as e:
            print(f"Warning: Could not extract subject ID from {file_path}: {e}")
            continue
    
    unique_subjects = list(subject_to_indices.keys())
    random.seed(random_seed)
    random.shuffle(unique_subjects)
    
    n_subjects = len(unique_subjects)
    n_train = int(splits[0] * n_subjects)
    n_val = int(splits[1] * n_subjects)
    
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for subject in train_subjects:
        train_indices.extend(subject_to_indices[subject])
    for subject in val_subjects:
        val_indices.extend(subject_to_indices[subject])
    for subject in test_subjects:
        test_indices.extend(subject_to_indices[subject])
    
    return train_indices, val_indices, test_indices

def get_demographic_stats():
    """
    Extract demographic statistics for train, validation, and test sets.
    """
    
    # Load participants data
    participants_df = pd.read_csv(path_participants, sep='\t')
    
    # Extract PTID (e.g., '002_S_0295') from participant_id (e.g., 'sub-ADNI002S0295')
    participants_df['PTID'] = participants_df['participant_id'].str.replace('sub-ADNI', '').str.replace('S', '_S_')
    
    # Get all file paths
    file_paths = find_pet(root_folder, prefix, extension, target_folder_name)
    
    # Split by subject ID
    train_indices, val_indices, test_indices = split_subjects_by_id(file_paths, splits)
    
    # Create dictionaries to map indices to file paths
    train_files = [file_paths[i] for i in train_indices]
    val_files = [file_paths[i] for i in val_indices]
    test_files = [file_paths[i] for i in test_indices]
    
    # Extract unique subject IDs for each split
    def get_unique_subjects(file_list):
        subjects = set()
        for file_path in file_list:
            try:
                id_ses = transform_string(file_path)
                subject_id = id_ses[0]  # PTID (XXX_S_XXXX)
                subjects.add(subject_id)
            except Exception as e:
                print(f"Warning: Could not extract subject ID from {file_path}: {e}")
                continue
        return list(subjects)
    
    train_subjects = get_unique_subjects(train_files)
    val_subjects = get_unique_subjects(val_files)
    test_subjects = get_unique_subjects(test_files)
    
    # Filter participants dataframe for each split
    train_participants = participants_df[participants_df['PTID'].isin(train_subjects)]
    val_participants = participants_df[participants_df['PTID'].isin(val_subjects)]
    test_participants = participants_df[participants_df['PTID'].isin(test_subjects)]
    
    # Calculate statistics for each split
    def calculate_stats(df, n_subjects, n_scans, split_name):
        # Age statistics
        age_mean = df['age_bl'].mean()
        age_std = df['age_bl'].std()
        age_min = df['age_bl'].min()
        age_max = df['age_bl'].max()
        
        # Sex distribution
        sex_counts = df['sex'].value_counts()
        n_male = sex_counts.get('M', 0)
        n_female = sex_counts.get('F', 0)
        pct_male = (n_male / n_subjects) * 100 if n_subjects > 0 else 0
        pct_female = (n_female / n_subjects) * 100 if n_subjects > 0 else 0
        
        return {
            'Split': split_name,
            '# Subjects': n_subjects,
            '# Scans': n_scans,
            'Age (mean ± std)': f'{age_mean:.1f} ± {age_std:.1f}',
            'Age Range': f'[{age_min:.1f}, {age_max:.1f}]',
            'Male (n, %)': f'{n_male} ({pct_male:.1f}%)',
            'Female (n, %)': f'{n_female} ({pct_female:.1f}%)'
        }
    
    # Gather statistics for all splits
    stats = []
    
    stats.append(calculate_stats(
        train_participants,
        len(train_subjects),
        len(train_files),
        'Training'
    ))
    
    stats.append(calculate_stats(
        val_participants,
        len(val_subjects),
        len(val_files),
        'Validation'
    ))
    
    stats.append(calculate_stats(
        test_participants,
        len(test_subjects),
        len(test_files),
        'Test'
    ))
    
    # Create DataFrame
    stats_df = pd.DataFrame(stats)
    
    return stats_df

def main():
    print("=" * 80)
    print("DEMOGRAPHIC STATISTICS FOR TRAIN/VALIDATION/TEST SPLITS")
    print("=" * 80)
    print()
    
    stats_df = get_demographic_stats()
    
    # Print table
    print(stats_df.to_string(index=False))
    print()
    
    # Calculate totals
    total_subjects = stats_df['# Subjects'].sum()
    total_scans = stats_df['# Scans'].sum()
    print(f"Total unique subjects: {total_subjects}")
    print(f"Total scans: {total_scans}")
    print(f"Average scans per subject: {total_scans/total_subjects:.2f}")
    print()
    
    # Save to CSV
    output_file = 'demographic_statistics.csv'
    stats_df.to_csv(output_file, index=False)
    print(f"Statistics saved to: {output_file}")
    print()
    
    # Print formatted table for easy copying
    print("=" * 80)
    print("FORMATTED TABLE (for easy copying to paper)")
    print("=" * 80)
    print()
    print("Split        | # Subjects | # Scans | Age (mean ± std) | Male n(%) | Female n(%)")
    print("-------------|------------|---------|------------------|-----------|-------------")
    for _, row in stats_df.iterrows():
        split_name = row['Split'].ljust(12)
        subjects = str(row['# Subjects']).rjust(10)
        scans = str(row['# Scans']).rjust(7)
        age = row['Age (mean ± std)'].rjust(16)
        male = row['Male (n, %)'].rjust(9)
        female = row['Female (n, %)'].rjust(11)
        print(f"{split_name} | {subjects} | {scans} | {age} | {male} | {female}")
    print()
    
    # Print LaTeX format for paper
    print("=" * 80)
    print("LATEX TABLE FORMAT")
    print("=" * 80)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Demographics of the dataset split}")
    print("\\label{tab:demographics}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("Split & \\# Subjects & \\# Scans & Age (mean $\\pm$ std) & Male n(\\%) & Female n(\\%) \\\\")
    print("\\hline")
    for _, row in stats_df.iterrows():
        male_str = row['Male (n, %)'].replace('%', '\\%')
        female_str = row['Female (n, %)'].replace('%', '\\%')
        print(f"{row['Split']} & {row['# Subjects']} & {row['# Scans']} & {row['Age (mean ± std)']} & {male_str} & {female_str} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

if __name__ == '__main__':
    main()
