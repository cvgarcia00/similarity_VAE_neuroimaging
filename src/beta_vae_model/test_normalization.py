import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from .normalisation import normalization_min, normalization_exp, normalization_hist


img = np.zeros((90, 110, 90))
img_file = nib.load('input_image.nii')
img[:, :-1, :] = img_file.get_fdata()[1:, :, 1:] # Skip first slice in each dimension to fit shape
img[np.isnan(img)] = 0 # Replace NaN values with 0

img_norm_exp = normalization_exp(img)
img_norm_hist = normalization_hist(img)
img_norm_min = normalization_min(img)

slice = 45

slice_img = img[:,:,45]
slice_norm_exp = img_norm_exp[:,:,45]
slice_norm_hist = img_norm_hist[:,:,45]
slice_norm_min = img_norm_min[:,:,45]

plt.imshow(slice_img, cmap='inferno')
plt.axis('off')  # Desactivar los ejes
plt.savefig('input_image.png', bbox_inches='tight', pad_inches=0)

plt.imshow(slice_norm_exp, cmap='inferno')
plt.axis('off')  # Desactivar los ejes
plt.savefig('normalization_exp.png', bbox_inches='tight', pad_inches=0)

plt.imshow(slice_norm_hist, cmap='inferno')
plt.axis('off')  # Desactivar los ejes
plt.savefig('normalization_hist.png', bbox_inches='tight', pad_inches=0)

plt.imshow(slice_norm_min, cmap='inferno')
plt.axis('off')  # Desactivar los ejes
plt.savefig('normalization_min.png', bbox_inches='tight', pad_inches=0)
