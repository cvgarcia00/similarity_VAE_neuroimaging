import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Union
import yaml

try:
    from config import load_config
except ImportError:
    from config import load_config


#___________________________CONFIG FILE__________________________

config_file = 'config.yaml'
config = load_config(config_file)
#LOAD HYPERPARAMETERS FROM CONFIG FILE

    
results_folder = 'results'

div_criteria = config['model']['divergence'] # Divergence function (KL, MMD)
latent = config['model']['latent'] # Latent space dimension
batch_size = config['model']['batch_size'] # Batch size
device = config['experiment']['device'] # Device to perform computation
recon_function = config['model']['reconstruction'] # Reconstruction function (MSE, DSSIM)
device_name = config['experiment']['device'] # Device name to perform computation
beta = config['model']['loss']['beta'] # Beta value for VAE
optimizer_model = config['model']['optimizer'] # Optimizer for training
learning_rate = config['model']['lr'] # Learning rate for training
correlation_type = config['model'].get('correlation_type', 'pearson') # Correlation type: pearson or spearman
spearman_temp = config['model'].get('spearman_temperature', 0.1) # Temperature for soft ranking in Spearman
folder = config['loader']['load_folder']['dataset'] # Folder to load NIFTI volumes
prefix = config['loader']['load_folder']['prefix'] # Prefix or files to be read
extension = config['loader']['load_folder']['extension'] # Extension of files to be read
target_folder_name = config['loader']['load_folder']['target_folder_name'] # Target folder name to be read (PET)
epochs = config['model']['epochs'] # Number of epochs to train  
splits = config['loader']['splits'] # Split rates for train, eval and test
path_ADNIMERGE = config['loader']['load_ADNIMERGE'] # Path to load ADNIMERGE csv file
normalization = config['experiment']['normalization'] # Normalization of images (TRUE or FALSE)




def MSE_3D(x: Tensor, y: Tensor, reduction: str = 'mean') -> Tensor:
    mse_loss = F.mse_loss(x, y, reduction=reduction)
    return mse_loss


def SSIM_3D(x: Tensor, y: Tensor, window_size: int = 5, reduction: str = 'mean', window_aggregation: str = 'mean') -> Union[Tensor, Tuple[Tensor, int]]:
    """ 
    Derived from: https://stackoverflow.com/questions/71357619/how-do-i-compute-batched-sample-covariance-in-pytorch
    Computes the Structural Similarity Index Measure (SSIM) for 3D images.

    Parameters:
    x (Tensor): A tensor representing the first batch of 3D images.
    y (Tensor): A tensor representing the second batch of 3D images.
    window_size (int): The size of the window to consider when computing the SSIM. Default is 5.
    reduction (str): The type of reduction to apply to the output: 'mean' | 'sum'. Default is 'mean'.
    window_aggregation (str): The type of aggregation to apply to the window: 'mean' | 'sum'. Default is 'mean'.

    Returns:
    Tensor: The SSIM value.
    int: The number of patches if reduction is 'sum'. Otherwise, returns None.
    """
    # Convert images to float and create patches
    patched_x = x.to(dtype=torch.float32).unfold(-3, window_size, window_size) \
        .unfold(-3, window_size, window_size) \
        .unfold(-3, window_size, window_size) \
        .reshape(x.shape[0], -1, window_size**3)
    patched_y = y.to(dtype=torch.float32).unfold(-3, window_size, window_size) \
        .unfold(-3, window_size, window_size) \
        .unfold(-3, window_size, window_size) \
        .reshape(y.shape[0], -1, window_size**3)

    # Compute statistics
    B, P, D = patched_x.size()
    varx, mux = torch.var_mean(patched_x, dim=-1)
    vary, muy = torch.var_mean(patched_y, dim=-1)
    diffx = (patched_x - mux.unsqueeze(-1)).reshape((B*P, -1))
    diffy = (patched_y - muy.unsqueeze(-1)).reshape((B*P, -1))
    covs = torch.bmm(diffx.unsqueeze(1), diffy.unsqueeze(2)).squeeze().reshape(B, P)/(D-1)

    # Compute SSIM
    c1, c2 = 0.01, 0.03
    numerador = (2*mux*muy + c1)*(2*covs + c2)
    denominador = (mux**2 + muy**2 + c1)*(varx + vary + c2)
    if window_aggregation == 'sum':
        ssim_bp = (numerador/denominador).sum(dim=-1)  # sum over windows
    elif window_aggregation == 'mean':
        ssim_bp = (numerador/denominador).mean(dim=-1)
    else:
        print(f'window reduction {window_aggregation} not supported')
        return None, None
    if reduction=='sum':
        raise ValueError(f'Window aggregation {window_aggregation} not supported')
    if reduction == 'sum':
        return ssim_bp.sum(), P
    else:
        return ssim_bp.mean(), None
    
def DSSIM_3D(x, y, window_size=5, reduction='mean', window_aggregation='mean'):
    ssim, P = SSIM_3D(x, y, window_size=window_size, reduction=reduction, window_aggregation=window_aggregation)
    if window_aggregation=='mean':
        P=1
    return (P-ssim)






class VAE_encoder(nn.Module):
    """
    Class for a 3D Encoder
    """
    
    def __init__(self, latent:int=10):
        """
        We define 2 convolutional layers, 1 fully connected layer
        and finally a fully connected layer for both mean and logvar
            Arg:
                - latent: dimension of the latent space
        We do not use pooling layers because stride=2 reduces the dimension
        """
        
        super().__init__()
        self.latent = latent
        #Input images have shape (90,110,90)
        
        
        self.conv = nn.Conv3d(1, 32, kernel_size= 11, stride = 3, padding = 1) #Output is (28, 34, 28)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv1 = nn.Conv3d(32 , 64, kernel_size = 7, stride = 2, padding = 1) #Output is (12, 15, 12)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64 , 128, kernel_size = 5, stride = 2, padding = 1) #Output is (5, 7, 5)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128 , 256, kernel_size = 3, stride = 2, padding = 1) #Output is (3, 4, 3)
        
        self.fc = nn.Linear(256 * 3 * 4 * 3  , 256) # Feat_maps * H * W * D 
        self.lat_mean = nn.Linear(256, self.latent)
        self.lat_logvar = nn.Linear(256, self.latent)
        
        
        
    def reparameterize(self, z_mean, z_logvar):
        """
        This function performs the reparameterization trick.
            Args:
                - z_mean: mean latent space tensor
                - z_logvar: logvar latent space tensor
                
            Output: z: reparameterized sample 
        """
        
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mean + std*eps
        return z
    
        
    def forward(self, x):
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0) #Add bath_size [bath_size, depth, height, width]    
        if len(x.shape) == 4:
            x = x.unsqueeze(1) #Reshapes x into [batch_size, 1, depth, height, width]
            
        x = F.relu(self.bn1(self.conv(x)))
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 256 * 3 * 4 * 3 ) #flatten data to fit linear layer
        x = F.relu(self.fc(x))
        
        z_mean = self.lat_mean(x)
        z_logvar = self.lat_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        return z, z_mean, z_logvar
    
    
    def divergence_loss_KL(self, z_mean, z_logvar):
        """
        Function for computing the KL divergence between
        two gaussians distributions. We assume priori distribution
        to be gaussian.
            Args:
                -z_mean: mean vector of latent space
                -z_logvar: logvar vector of latent space
                -beta: scalar to adjust the effect of divergence (Beta-VAE)
                
            Output:
                -KL_div_loss: KL divergence loss
        
        """
        
        KL_div_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
        KL_div_loss = torch.mean(KL_div_loss)
        
        return KL_div_loss
    
    
    def compute_kernel(self, x, y):
        """Function to compute gaussian kernel for MMD"""
        
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        
        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
        
        return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim = 2) / (dim*1.0))
    
    def compute_MMD(self, x:Tensor, y:Tensor):
        """
        Function for computing the MMD divergence between tensors x and y.
        """
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)
    
    def divergence_loss_MMD(self, z_sampled):
        """
        Function for computing MMD between sampled latent distribution and normal
        distribution.
        
        """
        #z = self.reparameterize(mu, logvar)        
        #z = (z - z.mean(dim = 0)) / (z.std(dim = 0) + 1e-7)
        z_prior = torch.randn(len(z_sampled), self.latent, device=device)
        
        return self.compute_MMD(z_prior, z_sampled)

    @staticmethod
    def _soft_rank(x, temperature=0.1):
        """
        Differentiable approximation of ranking using sigmoids.
        For each element x_i, its soft rank is: sum_j sigmoid((x_i - x_j) / temperature)
        As temperature -> 0, this converges to the true rank.
        
        Args:
            x: Tensor of shape [N, 1] or [N]
            temperature: Controls the smoothness of the approximation.
                         Lower = closer to true rank but harder gradients.
                         
        Returns:
            Tensor of soft ranks, same shape as input.
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        # x_i - x_j for all pairs: [N, N]
        pairwise_diff = x - x.T
        # Soft comparison via sigmoid
        soft_comparisons = torch.sigmoid(pairwise_diff / temperature)
        # Rank = sum of "how many elements is x_i greater than"
        ranks = soft_comparisons.sum(dim=1, keepdim=True)
        return ranks

    @staticmethod
    def _pearson_on_tensors(a, b):
        """
        Compute Pearson correlation between two tensors a and b.
        Both should be [N, 1] tensors (already filtered for NaN).
        
        Returns:
            pearson_corr: Tensor with the correlation value.
        """
        a = a - a.mean(dim=0, keepdim=True)
        b = b - b.mean(dim=0, keepdim=True)
        
        covar = torch.sum(a * b, dim=0)
        std_a = torch.sqrt(torch.sum(a ** 2, dim=0) + 1e-10)
        std_b = torch.sqrt(torch.sum(b ** 2, dim=0) + 1e-10)
        
        return covar / (std_a * std_b)

    #### PARA UN BETA COMPARTIDO ENTRE TARGETS

    def pearson_loss(self, latents, targets):
       
        # Use z_sampled instead of z_mean so that z_logvar is also trained
        # Filter NaN subjects
        
        
        if not isinstance(targets, dict):
            #Code if correlation is only between z0 and adas13
            adas13 = targets
            mask = ~torch.isnan(adas13)
            latents_filtered = latents[mask, 0:1] #only z0
            adas13 = adas13[mask].view(-1, 1)
            
            if latents_filtered.size(0) == 0:
                individual_losses = {'ADAS13': torch.zeros(1, device = latents.device)}
                individual_corrs = {'ADAS13': torch.zeros(1, device = latents.device)}
                return individual_losses, individual_corrs
            
            latents_filtered = latents_filtered - latents_filtered.mean(dim = 0, keepdim=True)
            adas13 = adas13 - adas13.mean(dim = 0, keepdim=True)
            
            covar = torch.sum(latents_filtered * adas13, dim = 0)
            std_latents = torch.sqrt(torch.sum(latents_filtered ** 2, dim = 0) + 1e-10)
            std_adas13 = torch.sqrt(torch.sum(adas13 ** 2, dim = 0) + 1e-10)
            
            pearson_corr = covar/ (std_latents * std_adas13)

            if torch.isnan(pearson_corr).any():
                # Devuelve diccionarios
                individual_losses = {'ADAS13': torch.zeros(1, device=latents.device)}
                individual_corrs = {'ADAS13': torch.zeros(1, device=latents.device)}
                return individual_losses, individual_corrs

            # Devuelve diccionarios
            individual_losses = {'ADAS13': -pearson_corr.mean()}
            individual_corrs = {'ADAS13': pearson_corr}
            return individual_losses, individual_corrs
            
        #Code for multiple targets
        individual_corrs = {}
        individual_losses = {}
        
        for idx, (key, target) in enumerate(targets.items()):
            #Filter NaN
            mask = ~torch.isnan(target)
            latents_filtered = latents[mask, idx:idx+1]   #This chooses the latent. idx = 0 for target 0, idx = 1 for target 1, etc
            target_filtered = target[mask].view(-1, 1)
            
            if latents_filtered.size(0) == 0:
                individual_losses[key] = torch.zeros(1, device = latents.device)
                individual_corrs[key] = torch.zeros(1, device = latents.device)
                continue
            
            #First, we normalize
            latents_filtered = latents_filtered - latents_filtered.mean(dim = 0, keepdim = True)
            target_filtered = target_filtered - target_filtered.mean(dim = 0, keepdim = True)
            
            #Compute correlation
            covar = torch.sum(latents_filtered * target_filtered, dim = 0)
            std_latents = torch.sqrt(torch.sum(latents_filtered ** 2, dim = 0) + 1e-10)
            std_target = torch.sqrt(torch.sum(target_filtered **2, dim = 0) + 1e-10)
            
            pearson_corr = covar / (std_latents * std_target)
            
            corr_loss = -pearson_corr.mean()
            
            if torch.isnan(pearson_corr).any():
                individual_losses[key] = torch.zeros(1, device = latents.device)
                individual_corrs[key] = torch.zeros(1, device = latents.device)
            else:
                individual_losses[key] = corr_loss
                individual_corrs[key] = pearson_corr
                
        
        return individual_losses, individual_corrs


    def spearman_loss(self, latents, targets, temperature=None):
        """
        Differentiable Spearman correlation loss.
        Identical logic to pearson_loss, but applies soft ranking to both
        latents and targets before computing Pearson correlation on the ranks.
        
        Spearman(x, y) = Pearson(rank(x), rank(y))
        
        Args:
            latents: Latent space tensor [batch, latent_dim]
            targets: Target tensor or dict of target tensors
            temperature: Temperature for soft ranking (lower = more accurate ranks
                         but potentially harder gradients). Default uses config value.
        """
        if temperature is None:
            temperature = spearman_temp
            
        if not isinstance(targets, dict):
            # Single target case (e.g., only ADAS13)
            adas13 = targets
            mask = ~torch.isnan(adas13)
            latents_filtered = latents[mask, 0:1]
            adas13 = adas13[mask].view(-1, 1)
            
            if latents_filtered.size(0) < 2:
                individual_losses = {'ADAS13': torch.zeros(1, device=latents.device)}
                individual_corrs = {'ADAS13': torch.zeros(1, device=latents.device)}
                return individual_losses, individual_corrs
            
            # Apply soft ranking
            ranked_latents = self._soft_rank(latents_filtered, temperature)
            ranked_target = self._soft_rank(adas13, temperature)
            
            # Pearson on ranks = Spearman
            pearson_corr = self._pearson_on_tensors(ranked_latents, ranked_target)
            
            if torch.isnan(pearson_corr).any():
                individual_losses = {'ADAS13': torch.zeros(1, device=latents.device)}
                individual_corrs = {'ADAS13': torch.zeros(1, device=latents.device)}
                return individual_losses, individual_corrs
            
            individual_losses = {'ADAS13': -pearson_corr.mean()}
            individual_corrs = {'ADAS13': pearson_corr}
            return individual_losses, individual_corrs
        
        # Multiple targets case
        individual_corrs = {}
        individual_losses = {}
        
        for idx, (key, target) in enumerate(targets.items()):
            mask = ~torch.isnan(target)
            latents_filtered = latents[mask, idx:idx+1]
            target_filtered = target[mask].view(-1, 1)
            
            if latents_filtered.size(0) < 2:
                individual_losses[key] = torch.zeros(1, device=latents.device)
                individual_corrs[key] = torch.zeros(1, device=latents.device)
                continue
            
            # Apply soft ranking
            ranked_latents = self._soft_rank(latents_filtered, temperature)
            ranked_target = self._soft_rank(target_filtered, temperature)
            
            # Pearson on ranks = Spearman
            pearson_corr = self._pearson_on_tensors(ranked_latents, ranked_target)
            
            corr_loss = -pearson_corr.mean()
            
            if torch.isnan(pearson_corr).any():
                individual_losses[key] = torch.zeros(1, device=latents.device)
                individual_corrs[key] = torch.zeros(1, device=latents.device)
            else:
                individual_losses[key] = corr_loss
                individual_corrs[key] = pearson_corr
        
        return individual_losses, individual_corrs


    def correlation_loss(self, latents, targets):
        """
        Dispatcher that selects between Pearson and Spearman correlation loss
        based on the config parameter 'correlation_type'.
        
        Args:
            latents: Latent space tensor
            targets: Target tensor(s)
            
        Returns:
            individual_losses, individual_corrs (same format as pearson_loss)
        """
        if correlation_type == 'spearman':
            return self.spearman_loss(latents, targets)
        else:
            return self.pearson_loss(latents, targets)


class VAE_decoder(nn.Module): 
    """
    Class for 3D VAE Decoder
    """
    
    def __init__(self, latent:int=10):
        super().__init__()
        self.latent = latent
        self.fc = nn.Linear(self.latent, 128 * 3 * 4 * 3)
        # out_padding to fit reconstructed dimensiones to real dimensions. Computed using W_out formula for Trans conv.
        self.conv1Trans = nn.ConvTranspose3d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = (1, 0, 1)) 
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2Trans = nn.ConvTranspose3d(64, 32, kernel_size = 4, stride = 3, padding = 1, output_padding = (0, 1, 0))
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3Trans = nn.ConvTranspose3d(32 , 1, kernel_size = 11, stride = 5, padding = 1, output_padding = (1, 1, 1))
    
        
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 128, 3, 4, 3) #This rearranges shapes in volumes
        x = F.relu(self.bn1(self.conv1Trans(x)))
        x = F.relu(self.bn2(self.conv2Trans(x)))
        x = F.relu(self.conv3Trans(x))
        
        return x
    

    def loss_recon(self, x_target, x_recon):
        """
        Function for computing the loss due to reconstruction of 3D volumes.
        We use DSSIM defined as 1-SSIM.
            Args:
                -x: input data. (3D volumes)
                -x_recon: reconstructed data. (3D volumes)
                
                
            Output:
                -loss: reconstruction loss computed with DSSIM
        
        """
        
        recon_loss = DSSIM_3D(x_target, x_recon)
        return recon_loss





class VAE(nn.Module):
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    
    def forward(self, x):
        z, z_mean, z_logvar = self.encoder(x)
        
        x_recon = self.decoder(z)
        
        return x_recon, z, z_mean, z_logvar
    
    
    def loss(self, x_target, x_recon, z_sampled, z_mean, z_logvar, beta_corr_dict, targets = None):
        
        # Compute correlation between target and latents
        if targets is not None:
            individual_losses, individual_corrs = self.encoder.correlation_loss(z_sampled, targets)
            
            corr_loss = 0.0
            for key, loss_val in individual_losses.items():
                corr_loss += beta_corr_dict.get(key, 0.0) * loss_val
            
        else:
            corr_loss = torch.tensor(0.0, device=x_target.device)
            individual_corrs = torch.tensor(0.0, device = x_target.device)    
        
        # Compute reconstruction loss
        if recon_function == 'MSE':
            recon_loss = MSE_3D(x_target, x_recon)
        elif recon_function == 'DDSIM':
            recon_loss = self.decoder.loss_recon(x_target, x_recon)

        # Compute divergence loss
        if div_criteria == 'KL':
            div_loss = self.encoder.divergence_loss_KL(z_mean, z_logvar)
            # If KL --> + sign in divergence loss
            total_loss = recon_loss + beta*div_loss + corr_loss
        elif div_criteria == 'MMD':
            div_loss = self.encoder.divergence_loss_MMD(z_sampled)
            # If MMD --> - sign in divergence loss (InfoVAE)
            #Minimizing corr_loss = -pearson_corr is equivalent to maximizing pearson_corr
            total_loss = recon_loss - beta * div_loss + corr_loss
        
  
        return  total_loss, beta * div_loss, recon_loss, individual_corrs, corr_loss








# # #################### CONDITIONAL VAE  ####################
# """
# CVAE IS EXACTLY LIKE VAE BUT WE CONCATENATE BIOMARKERS TO THE LINEAR LAYER OF IMAGES PREVIOUS
# TO THE LATENT SPACE. Biomarkers_num is the number of biomarkers we are conditioning the model with.
# Default is set to 1, ADAS13.
# """

# class CVAE_encoder(VAE_encoder):
#     def __init__(self, latent_dim: int = 10, biomarkers_num: int  = 1):
#         super().__init__()
#         self.latent = latent_dim
#         self.biomarkers_num =  biomarkers_num
        
#         self.conv = nn.Conv3d(1, 32, kernel_size= 11, stride = 3, padding = 1) #Output is (28, 34, 28)
#         self.bn1 = nn.BatchNorm3d(32)
#         self.conv1 = nn.Conv3d(32 , 64, kernel_size = 7, stride = 2, padding = 1) #Output is (12, 15, 12)
#         self.bn2 = nn.BatchNorm3d(64)
#         self.conv2 = nn.Conv3d(64 , 128, kernel_size = 5, stride = 2, padding = 1) #Output is (5, 7, 5)
#         self.bn3 = nn.BatchNorm3d(128)
#         self.conv3 = nn.Conv3d(128 , 256, kernel_size = 3, stride = 2, padding = 1) #Output is (3, 4, 3)
    
        
#         self.fc_img = nn.Linear(256 * 3 * 4 * 3, 128) # Flattened size from convolutional layers. Image features
        
#         self.lat_mean = nn.Linear(128 + self.biomarkers_num + 1, latent) # Mean of latent space (+1 to account for NaN tags)
#         self.lat_logvar = nn.Linear(128 + self.biomarkers_num + 1, latent) # Logvar of latent space

    
#     def forward(self, x, biomarkers):
#         """
#         Forward pass through the encoder.
#         Args:
#             - x: input image tensor of shape [batch_size, 1, depth, height, width]
#             - biomarkers: input biomarker tensor of shape [batch_size, biomarkers_num]
#         """

#         if len(x.shape) == 3:
#             x = x.unsqueeze(0)  # Add batch_size [batch_size, depth, height, width]
#         if len(x.shape) == 4:
#             x = x.unsqueeze(1) # Reshapes x into [batch_size, 1, depth, height, width]
                
#         x = F.relu(self.bn1(self.conv(x)))
#         x = F.relu(self.bn2(self.conv1(x)))
#         x = F.relu(self.bn3(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 256 * 3 * 4 * 3 ) #flatten data to fit linear layer
#         x = F.relu(self.fc_img(x))
        
#         biomarkers = biomarkers.float()
#         biomarkers = biomarkers.T
#         x = torch.cat([x, biomarkers], dim = 1) # Concatenate image features with biomarkers
        
        
#         z_mean = self.lat_mean(x)
#         z_logvar = self.lat_logvar(x)
#         z = self.reparameterize(z_mean, z_logvar)
#         return z, z_mean, z_logvar
    
    
    
    
# class CVAE_decoder(VAE_decoder):
#     def __init__(self, latent_dim: int = 10, biomarkers_num: int = 2):
#         super().__init__()
#         self.latent = latent_dim
#         self.biomarkers_num =  biomarkers_num
        
#         self.fc_latent = nn.Linear(self.latent, 256 + self.biomarkers_num)
#         self.fc2 = nn.Linear(256 + self.biomarkers_num, 128 * 3 * 4 * 3)
#         # out_padding to fit reconstructed dimensiones to real dimensions. Computed using W_out formula for Trans conv.
#         self.conv1Trans = nn.ConvTranspose3d(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = (1, 0, 1)) 
#         self.conv2Trans = nn.ConvTranspose3d(64, 32, kernel_size = 4, stride = 3, padding = 1, output_padding = (0, 1, 0))
#         self.conv3Trans = nn.ConvTranspose3d(32 , 1, kernel_size = 11, stride = 5, padding = 1, output_padding = (1, 1, 1))


        
#     def forward(self, z):
#         """
#         Forward pass through the decoder.
#         Args:
#             - z: latent space tensor of shape [batch_size, latent]
#             - biomarkers: biomarker tensor of shape [batch_size, biomarkers_num]
#         """
        
#         x = F.relu(self.fc_latent(z))
#         x = F.relu(self.fc2(x))
#         x = x.view(-1, 128, 3, 4, 3) # Reshape to match output
        
#         # Decode image
        
#         x = F.relu(self.conv1Trans(x))
#         x = F.relu(self.conv2Trans(x))
#         x = F.relu(self.conv3Trans(x))
        
#         return x
    
    
    
# class CVAE(VAE):
#     def __init__(self, encoder, decoder):
#         """
#         Conditional Variational Autoencoder (CVAE) model.
#         Args:
#             - latent: dimension of the latent space
#             - biomarkers_num: dimension of the biomarker vector
#         """
#         super().__init__(encoder, decoder)
#         self.encoder = encoder
#         self.decoder = decoder
            
#     def forward(self, x, biomarkers):
            
#         # Encode image and biomarkers
#         z, z_mean, z_logvar = self.encoder(x, biomarkers)
            
#         # Decode
#         recon_x = self.decoder(z)
            
#         return recon_x, z, z_mean, z_logvar
        