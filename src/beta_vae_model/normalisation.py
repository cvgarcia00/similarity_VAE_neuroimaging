import yaml
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import nibabel as nib 
from skimage import exposure
from config import load_config

#from tqdm import tqdm
#from scipy.ndimage import binary_erosion, binary_dilation
#from skimage import filters


config_file = 'config.yaml'
config = load_config(config_file)

path = config['loader']['load_temp']

#_______________________________________

#       NORMALIZATION FUNCTIONS
#_______________________________________


def normalization_hist(img):
    img_int = np.maximum(0, img)
    
    img_norm = exposure.equalize_hist(img_int)
    # p2, p98 = np.percentile(img, (2, 98))
    # img_norm = exposure.rescale_intensity(img_int, in_range=(p2, p98))
    img_norm = exposure.equalize_adapthist(img_norm, clip_limit=0.01)
    return img_norm


def normalization_min(img):
    """
    For dataloader we do not need the img_norm_list because we normalize each 
    image independently.
    """
    img_flat = img.flatten()
    max_voxels = np.nanpercentile(img_flat, 99)
    mean_max_voxels = np.nanmean(img_flat[img_flat>=max_voxels])
    img_norm = np.minimum(1, img/mean_max_voxels)
    img_norm = np.maximum(0, img_norm)
              
    return img_norm

def normalization_max(img):
    max_value = np.nanmax(img)
    img_norm = np.minimum(1, img/max_value)
    img_norm = np.maximum(0, img_norm)
              
    return img_norm

def normalization_exp(img):
    img_min = normalization_min(img)
    img_norm = (np.exp(img_min)-1)/np.exp(1)
    return img_norm



# def normalization_min(img_list):
#     """
#     This function performs the normalization of a list of images.
#     Normalization is performed using the mean of the 5% of voxels with
#     more intensity. We delimit by 1 using minimum between normalized voxel
#     and 1.
#     """
#     img_norm_list = []
    
#     for img in img_list:
#         img_flat = img.flatten()
#         max_voxels = np.nanpercentile(img_flat, 95)
#         mean_max_voxels = np.nanmean(img_flat[img_flat>=max_voxels])
#         img_norm = np.minimum(1, img/mean_max_voxels)
    
#         img_norm_list.append(img_norm)
             
#     return img_norm_list



def normalization_cerebellum(img):
    """
    Function for loading a template from AAL3 (ROI_MNI_V7.nii).
    It extracts the cerebellum region and load it into an array.
    
    Normalization is performed by applying cerebellum template as
    a mask to an image, performing intensity average of cerebellum region
    and dividing volume by average.
    
    This is done because cerebellum is NOT affected by Alzheimer's disease.
    
    Volumes must be corregistered to single_subj_T1.nii MNI space as
    indicated in the original AAL article.
        Args:
            -img_list: List of volumes (corregistered) to normalize
            
        Output:
            -img_list_norm: List of volumes normalized in intensity
            according to cerebellum average 
    
    """

    temp_file = nib.load(path)
    temp = np.zeros((90,110,90))
    #We use 1:, :-1, and 1: because we want shape (90, 110, 90) instead of (91, 109, 91)
    temp[:,:-1,:] = temp_file.get_fdata()[1:,:,1:]
    
    #Set everything to 0 except for cerebellum labels (from 95 to 120)
    temp_cerebellum = np.zeros_like(temp)
    
    temp_cerebellum[np.logical_and(temp >= 95, temp <= 120)] = temp[np.logical_and(temp >= 95, temp <= 120)]
    #Transform to binary to obtain mask
    mask = np.zeros_like(temp_cerebellum)
    mask[temp_cerebellum != 0] = 1.0
    
    img_list_norm = []
    for img in img_list:
        #img_masked is a volume which ONLY contains the cerebellum of a patient
        img_masked = np.multiply(mask, img)
        #Get nonzero values of img once mask is applied (to not include 0s in average)
        nonzero_img_norm = img_masked[img_masked != 0] 
        #Compute average of cerebellum, normalise and add to list. Nanmean to avoid NaN values
        norm = np.nanmean(nonzero_img_norm)
        #print(f'norm of iter {i} is: {norm}')
        img_norm = img / norm
        img_list_norm.append(img_norm)
    
    
    return img_list_norm













