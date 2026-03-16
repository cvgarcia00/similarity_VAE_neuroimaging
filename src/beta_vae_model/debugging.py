import numpy as np
import os
import nibabel as nib
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import load_config
from pathlib import Path

config_file = 'config.yaml'
config = load_config(config_file)

#We load the saving folders relative to root folder
root_folder = config["root"]
root_folder = Path(root_folder) #Converts string to path 
results_folder = root_folder / config["folders"]["results"]

def make_grid_recon(batch, string):
    
    slice_index = 45 #SLICE OF VOLUME
    
    if len(batch.shape) == 4:
        batch = batch.unsqueeze(1) #If input is [BATCH_SIZE, HEIGHT, WIDTH, DEPTH] add channel dimension
    
    batch_slices = batch[:, :, :, :, slice_index] #Output is [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
    
    batch_size = batch_slices.size(0)
    nrow = int(np.sqrt(batch_size))

    
    grid = make_grid(batch_slices, nrow = nrow)
    
    os.makedirs(results_folder, exist_ok=True)
    
    grid_normalized = (grid - grid.min()) / (grid.max() - grid.min())
    
    file_path = os.path.join(results_folder, f'grid_images{string}.png')
    plt.imsave(file_path, grid_normalized.permute(1, 2, 0).cpu().numpy()[...,0], vmin=0, vmax=1, cmap='inferno')
    return None


def tensor_to_nii(x_input, x_recon):
    """
    This function transforms an input volume tensor 'x_input' into a .nii
    volume 'x_recon'
    """
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    x_recon = x_recon.squeeze(0) #Erase channel dimension
    x_numpy = x_recon.cpu().detach().numpy()
    x_nifti_recon = nib.Nifti1Image(x_numpy, affine = np.eye(4))
    
    x_input = x_input.squeeze(0)
    x_input = x_input.cpu().detach().numpy()
    x_nifti_input = nib.Nifti1Image(x_input, affine = np.eye(4))
    nib.save(x_nifti_recon, os.path.join(results_folder, 'reconstructed_test_volume.nii'))
    nib.save(x_nifti_input, os.path.join(results_folder, 'normalised_test_volume.nii'))
    return 



def reconstruction_diff(x, y):
    """
    This function measures the reconstruction difference between two volumes.
    Difference is performed by direct substraction of pixels.
    """
    
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    x = x.squeeze(0) #Erase channel dimension (We need 3dims)
    y = y.squeeze(0) #Erase channel dimension (We need 3dims)
    recon_img = x - y
    recon_img_np = recon_img.detach().cpu().numpy()
    nii_recon_diff = nib.Nifti1Image(recon_img_np, affine = np.eye(4))
    nib.save(nii_recon_diff, os.path.join(results_folder, 'ImageDiff.nii'))
    return recon_img

def get_ADNI_BIDS_HIST(id_ses_list_formated, df_ADNIMERGE, feature):
    
    #GET CSV OF ONLY ADNI_BIDS
    df_id_ses = pd.DataFrame(id_ses_list_formated, columns = ['PTID', 'VISCODE'])
    df_ADNI_BIDS = pd.merge(df_ADNIMERGE, df_id_ses, on = ['PTID', 'VISCODE'])
    #df_ADNI_BIDS.to_csv('ADNI_BIDS.csv', index = False)
    
    #GET ADAS13 HISTOGRAM
    plt.figure(figsize=(8,6))
    plt.hist(df_ADNI_BIDS[f'{feature}'], bins = 50, color = 'skyblue', edgecolor = 'black')
    plt.xlabel(f'{feature}')
    plt.ylabel('Count')
    
    plt.savefig(f'{feature}_ADNI_BIDS_HIST.png', dpi=300, bbox_inches='tight')
    plt.close()
    return df_ADNI_BIDS






def plot_distribution(z_mean_list, z_logvar_list):
    
    z_mean = np.vstack([tensor.cpu().numpy() for tensor in z_mean_list])  # Stack in a 2D array
    z_logvar = np.vstack([tensor.cpu().numpy() for tensor in z_logvar_list])
    
    
    distribution_folder = os.path.join(results_folder, 'distribution_folder')
    
    if not os.path.exists(distribution_folder):
        os.makedirs(distribution_folder)
    
    latent_dim = z_mean.shape[1]
    
    #FOR DENSITY OF Z_MEAN
    
    fig, axes = plt.subplots(latent_dim, 1, figsize = (10, latent_dim*3))
    
    for i in range(latent_dim):
        sns.kdeplot(z_mean[:, i], fill = True, color = 'g', ax = axes[i])
        axes[i].set_title(f'Probability density of z_mean for dimension {i}')
        axes[i].set_xlabel('latent variable {i}')
        axes[i].set_ylabel('Density')
        
    plt.tight_layout()
    
    
    output_filename = 'distributions_zmean.png'  
    save_path = os.path.join(distribution_folder, output_filename)
    plt.savefig(save_path)
    
    plt.show()
    
    #_________________________________
    
    #FOR DENSITY OF Z_LOGVAR
    
    fig, axes = plt.subplots(latent_dim, 1, figsize = (10, latent_dim*3))
    
    for i in range(latent_dim):
        sns.kdeplot(z_logvar[:, i], fill = True, color = 'g', ax = axes[i])
        axes[i].set_title(f'Probability density of z_logvar for dimension {i}')
        axes[i].set_xlabel('latent variable {i}')
        axes[i].set_ylabel('Density')
        
    plt.tight_layout()
    
    output_filename = 'distributions_z_logvar.png'  
    save_path = os.path.join(distribution_folder, output_filename)
    plt.savefig(save_path)
    
    plt.show()
    
    #FOR DENSITY OF BOTH IN THE SAME IMAGE
    
    fig, axes = plt.subplots(latent_dim, 1, figsize=(10, latent_dim * 4))
    
    
    for i in range(latent_dim):
        
        ax = axes[i]
        sns.kdeplot(z_mean[:, i], label=f'z_mean {i}', color='blue', fill=True, alpha=0.5, ax=ax)
    
        # TO COMPARE WE TRANSFORM LOGVAR INTO SIGMA
        sigma = np.exp(0.5 * z_logvar[:, i])
        sns.kdeplot(sigma, label=f'z_logvar {i}', color='red', fill=True, alpha=0.5, ax=ax)
    
        ax.set_title(f'Densities of z_mean and sigma for latent {i}')
        ax.set_xlabel('Latent variable {i}')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    
    output_filename = 'distributions_zmean_sigma.png'  
    save_path = os.path.join(distribution_folder, output_filename)
    plt.savefig(save_path)
    
    plt.show()
    
    return None


