from .debugging import make_grid_recon
from .plot_results import SVR_feature, plot_latentx_vs_feature, plot_latent_space_feature, global_glm_model, glm_model, tsne_latent
from .plot_results import  class_AD_vs_CN, class_AD_vs_CN_ADAS, class_AD_vs_CN_First_Latent
from .VAE_model_review import VAE, VAE_encoder, VAE_decoder
from .dataloader import BrainDataset, get_dataloader, verify_subject_separation
import os
import sys
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from colorama import Fore, init
import shutil
import pandas as pd
from config import load_config
import numpy as np
from pathlib import Path


#___________________________CONFIG FILE__________________________


config_file = 'config.yaml'
config = load_config(config_file)

#LOAD HYPERPARAMETERS FROM CONFIG FILE

# root_folder = config["root"]
# root_folder = Path(root_folder) #Converts string to path 
# results_folder = root_folder / config["folders"]["results"]
# shutil.copy(config_file, results_folder)

div_criteria = config['model']['divergence'] # Divergence function (KL, MMD)
latent = config['model']['latent'] # Latent space dimension
batch_size = config['model']['batch_size'] # Batch size
device = config['experiment']['device'] # Device to perform computation
recon_function = config['model']['reconstruction'] # Reconstruction function (MSE, DSSIM)
device_name = config['experiment']['device'] # Device name to perform computation
beta = config['model']['loss']['beta'] # Beta value for VAE
optimizer_model = config['model']['optimizer'] # Optimizer for training
learning_rate = config['model']['lr'] # Learning rate for training
folder = config['loader']['load_folder']['dataset'] # Folder to load NIFTI volumes
prefix = config['loader']['load_folder']['prefix'] # Prefix or files to be read
extension = config['loader']['load_folder']['extension'] # Extension of files to be read
target_folder_name = config['loader']['load_folder']['target_folder_name'] # Target folder name to be read (PET)
epochs = config['model']['epochs'] # Number of epochs to train  
splits = config['loader']['splits'] # Split rates for train, eval and test
path_ADNIMERGE = config['loader']['load_ADNIMERGE'] # Path to load ADNIMERGE csv file
normalization = config['experiment']['normalization'] # Normalization of images (TRUE or FALSE)
architecture = config['model']['architecture'] # Architecture of the model (VAE, CVAE)

correlation_targets = config['model']['correlation_targets']
beta_corr_weights = config['model']['loss']['beta_corr_weights']

#___________________SET OF FEATURES TO BE USED____________________

feature_ADAS = config['results']['feature_ADAS']
feature_Ventricles = config['results']['feature_Ventricles']
feature_Hippo = config['results']['feature_Hippo']
feature_MidTemp = config['results']['feature_MidTemp']    
feature_Fusiform = config['results']['feature_Fusiform']
feature_Entorhinal = config['results']['feature_Entorhinal']
feature_PTAU = config['results']['feature_PTAU']
feature_ABETA = config['results']['feature_ABETA']
feature_MMSE = config['results']['feature_MMSE']
feature_CDRSB = config['results']['feature_CDRSB']
feature_AGE = config['results']['feature_AGE']
    
base_features = ["ABETA", "PTAU", "ADAS13", "Ventricles", 
                 "Hippocampus", "Entorhinal", "Fusiform", "MidTemp", 'MMSE', 'DX']

#Set eliminates duplicates
feature_list_ADNI_BIDS = list(set(base_features + correlation_targets))

# THESE ARE THE FEATURES USED FOR PERFORMING ANALYSIS
features = [
    (config_key, score_name) 
    for config_key, score_name in config['results'].items()
    if config_key.startswith('feature_')
]


class EarlyStopping:
    """
    This class performs early stopping during training. It saves the best model with a patience (number of epochs to wait).
    """
    def __init__(self, results_folder, patience=100, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta # Minimum improved required to continue
        self.save_path = os.path.join(results_folder, 'vae_model.pth')
        
    def __call__(self, val_loss, model, optimizer, epoch, loss_train, loss_eval, loss_eval_div, loss_eval_recon, latent_dim, results_folder):
        #If no better loss, update
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_model(model, optimizer, epoch, loss_train, loss_eval, loss_eval_div, loss_eval_recon, latent_dim, results_folder)
        else:
            self.counter +=1
            if self.counter >= self.patience:
                self.early_stop = True
                
    def save_model(self, model, optimizer, epoch, loss_train, loss_eval, loss_eval_div, loss_eval_recon, latent_dim, results_folder):
        
        with open(os.path.join(results_folder, 'best_loss_eval.txt'), 'w') as file:
            file.write(f'Best_loss_eval: {self.best_loss}, Epoch: {epoch}\n')
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_eval': loss_eval,
            'loss_eval_div': loss_eval_div,
            'loss_eval_recon': loss_eval_recon,
            'latent_dim': latent_dim
        }, self.save_path)
        
        print(f'Model saved')


"""
Choose the model. Right now we only have VAE architecture.
"""

def model_VAE():
    if architecture == 'VAE':
        model = VAE(encoder = VAE_encoder(latent),
                    decoder = VAE_decoder(latent))
        return model
    else:
        raise Exception("Incorrect model type")

#_________________________________________________

#               TRAINING
#_________________________________________________


def epoch_train(model, train_loader, optimizer, epoch, beta_corr_dict, writer, results_folder):
    """
    This function performs training of the model for an epoch. It computes the loss of each batch 
    and updates the weights
        Args:
            -model: deep network model.
            -train_loader: dataloader of training. Contains images and features.
            -optimizer: optimizer for training. (e.g., Adam)
            -epoch: number of training epochs.
            -beta_corr: regularization parameter of the similarity loss.
            -writer: to keep track of saved data.
        
        Output:
            - Loss of an epoch.
    
    """ 

    
    model.train()
    loss_epoch = 0.0
    loss_epoch_div = 0.0
    loss_epoch_recon = 0.0
    loss_epoch_corr = 0.0
    batch_count = 0
    
    avg_individual_corrs = {target: 0.0 for target in correlation_targets}
    
    if correlation_targets:
        first_target_name = correlation_targets[0]
    else:
        raise ValueError("No targets in correlation targets")
    
    use_multi_target = epoch > epochs//3
    if not use_multi_target:
        avg_pearson = 0.0
    
    for batch_img, train_dict in train_loader:
        
        batch = batch_img.to(device)   
        optimizer.zero_grad()   #Erase gradients from previous step
        
        if use_multi_target:
            targets = {
                target: train_dict[target].to(device)
                for target in correlation_targets
            }
        else:
            targets = train_dict[first_target_name].to(device)
            
        x_recon_batch, z_sampled, z_mean, z_logvar = model(batch)
        
        # Remove channel dimension for loss computation
        
        if len(x_recon_batch.shape) == 5: # ndim
            x_recon_batch = x_recon_batch.squeeze(1) # [BATCH, DIMS]
        
        if len(batch.shape) == 5:
            batch = batch.squeeze(1)
         
        # Compute the loss :D
        loss_batch, loss_train_div, loss_train_recon, individual_corrs, corr_loss_train = model.loss(batch, x_recon_batch, z_sampled, z_mean, z_logvar, beta_corr_dict, targets)
        
        # Backpropagate the loss of the batch
        loss_batch.backward()
        optimizer.step()
        
        # Save the interesting data and update the writer to keep track
        loss_epoch += loss_batch.item()
        loss_epoch_div += loss_train_div.item()
        loss_epoch_recon += loss_train_recon.item()
        loss_epoch_corr += corr_loss_train.item()
        
        if isinstance(individual_corrs, dict):
            for key, corr in individual_corrs.items():
                avg_individual_corrs[key] += corr.item() if hasattr(corr, 'item') else corr
        else:
            avg_pearson += individual_corrs.item() if hasattr(individual_corrs, 'item') else individual_corrs
        
        if (epoch % max(1, epochs // 8) == 0 and batch_count == 0) :      #Save images every epoch/X to check training progress
            make_grid_recon(x_recon_batch, string = f'train_recon_epoch_{epoch}')
            
            x_recon_batch = x_recon_batch.unsqueeze(1)
            string_1 = f'train_recon_epoch'
            writer.add_images(string_1, x_recon_batch[:, :, :, :, 45], epoch)
            
            # batch = batch.unsqueeze(1)
            make_grid_recon(batch, string = f'train_input_epoch_{epoch}')  
            
            batch = batch.unsqueeze(1)
            string_2 = f'train_input_epoch'
            writer.add_images(string_2, batch[:, :, :, :, 45], epoch)
            
            df = pd.DataFrame(z_mean.detach().cpu().numpy(), columns = [f'z_{i}' for i in range(latent)])
            df[first_target_name] = train_dict[first_target_name]
            
            df.to_csv(os.path.join(results_folder, f'z_mean_{first_target_name}_epoch_{epoch}.csv'))
            
            
            
        batch_count += 1 
        
    #Average over total number of batches
    loss_epoch /= batch_count
    loss_epoch_div /= batch_count
    loss_epoch_recon /= batch_count
    loss_epoch_corr /= batch_count
    
    if use_multi_target:
        for key in avg_individual_corrs:
            avg_individual_corrs[key] = -avg_individual_corrs[key] / batch_count
        return loss_epoch, loss_epoch_div, loss_epoch_recon, loss_epoch_corr, avg_individual_corrs
    else:
        avg_pearson = -avg_pearson / batch_count
        return loss_epoch, loss_epoch_div, loss_epoch_recon, loss_epoch_corr, avg_pearson






def train(model, train_set, eval_set, early_stopping, results_folder):
    
    writer = SummaryWriter() #Keep track of data through tensorboard
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    
    #Load model into cuda device and choose optimizer
    model = model.to(device)
    if optimizer_model == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    elif optimizer_model == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
    else: raise ValueError("Invalid optimizer")

    for epoch in range(epochs):
        #Train the model for this epoch
        print(f'{Fore.YELLOW} Epoch nº: {epoch}')
        
        # Scheduler for similarity regularization weight
        """
        The idea is to train the model using only RECON + KL_DIV for a certain number of epochs,
        so that the model captures patterns from the scans. After those epochs, we introduce the
        similarity regularization to guide the model to align to dementia score.
        """
        if epoch <= epochs // 3:
            beta_corr_dict = {target: 0.0 for target in correlation_targets}
        else: 
            beta_corr_dict = {target: beta_corr_weights[target] for target in correlation_targets}
        print(f'Value of beta_corr_dict = {beta_corr_dict}')
        
        
        #Train
        loss_epoch, loss_epoch_div, loss_epoch_recon, loss_epoch_corr, corr_metrics = \
            epoch_train(model, train_set, optimizer, epoch, beta_corr_dict, writer, results_folder)
        
        #Evaluate
        loss_eval_epoch, loss_eval_div, loss_eval_recon, eval_corr_metrics, mean_distance_eval, var_eval_phase_diag = \
            evaluation(model, eval_set, epoch, beta_corr_dict)
        
        #______________________KEEP TRACK WITH TENSORBOARD_____________________
        
        #Write train loss of each epoch and its divergence/recon loss
        writer.add_scalar('Loss/train', loss_epoch, epoch)
        writer.add_scalars('Loss_train/div-recon', 
            {'loss train div' : loss_epoch_div, 
             'loss train recon' : loss_epoch_recon, 
             'loss train correlation': loss_epoch_corr},
        epoch)
        
        #Write evaluation loss of each epoch and its divergence/recon loss 
        writer.add_scalar('Loss/evaluation', loss_eval_epoch, epoch)
        writer.add_scalars('Loss_evaluation/div_recon', 
            {'loss eval div' : loss_eval_div, 
             'loss eval recon': loss_eval_recon}, 
        epoch)
        
        #Correlations
        if isinstance(corr_metrics, dict):
            for key, val in corr_metrics.items():
                writer.add_scalar(f'Pearson correlation - training/{key}', val, epoch)
            for key, val in eval_corr_metrics.items():
                writer.add_scalar(f'Pearson correlation - validation/{key}', val, epoch)
        else:
            writer.add_scalar('Pearson correlation - training/ADAS13', corr_metrics, epoch)
            writer.add_scalar('Pearson correlation - validation/ADAS13', eval_corr_metrics, epoch)
            
        #Print some general information
        print(f'Epoch loss training: {loss_epoch}, validation loss: {loss_eval_epoch}')
        print(f'Correlation metrics train: {corr_metrics}')
        print(f'Correlation metrics eval: {eval_corr_metrics}')
        
        
        # Early stopping and save best model

        early_stopping(val_loss=loss_eval_epoch, 
                       model=model, 
                       optimizer=optimizer, 
                       epoch=epoch, 
                       loss_train=loss_epoch, 
                       loss_eval=loss_eval_epoch, 
                       loss_eval_div=loss_eval_div, 
                       loss_eval_recon=loss_eval_recon, 
                       latent_dim=latent, 
                       results_folder=results_folder)
        
        if early_stopping.early_stop:
            print('Early stopping activated. Finishing training...')
            break

    return model

#______________________________________

#           EVALUATION
#______________________________________


def evaluation(model, eval_loader, epoch, beta_corr_dict):
    model.eval()
    
    count = 0
    loss_eval = 0.0
    loss_eval_div = 0.0
    loss_eval_recon = 0.0
    avg_individual_corrs = {'ADAS13': 0.0, 'CDRSB': 0.0}
    
    use_multi_target = epoch > epochs // 3
        
    all_z_means = []
    
    with torch.no_grad():
        for x, features_dict in eval_loader:
            
            x = x.to(device)
            
            if use_multi_target:
                targets = {
                    'ADAS13': features_dict['ADAS13'].to(device),
                    'CDRSB': features_dict['CDRSB'].to(device)
                }
            else:
                targets = features_dict['ADAS13'].to(device)
            
            recon_eval, z_sampled, z_mean, z_logvar = model(x)
            
                
            if len(recon_eval.shape) == 5: # ndim
                recon_eval = recon_eval.squeeze(1) # [BATCH, DIMS]
            if len(x.shape) == 5:
                x = x.squeeze(1)
                
            loss_eval_batch , loss_eval_div_batch, loss_eval_recon_batch, individual_corrs, _ = model.loss(x, recon_eval, z_sampled, z_mean, z_logvar, beta_corr_dict, targets)
                
            loss_eval += loss_eval_batch.item()
            loss_eval_div += loss_eval_div_batch.item()
            loss_eval_recon += loss_eval_recon_batch.item()
            

            for key, val in individual_corrs.items():
                if key in avg_individual_corrs:
                    avg_individual_corrs[key] += val.item() if hasattr(val, 'item') else val

            
            count += 1
            all_z_means.append(z_mean.detach().cpu())
            
            
        all_z_means_tensor = torch.cat(all_z_means, dim = 0) #[N, latent_dim]
        mean_latent = all_z_means_tensor.mean(dim = 0) # Mean of latent vectors
        
        # We compute the variance of the means of one dimension across all subjects
        # Low variance -> model is not capturing variance.
        var_latent = all_z_means_tensor.var(dim = 0)
        var_latent_mean = var_latent.mean().item()
        
        distances_to_mean = torch.norm(all_z_means_tensor - mean_latent, dim = 1) #Euclidean distance of each point to the mean
        mean_distance_to_mean = distances_to_mean.mean().item() #Mean distance to the mean
            
        loss_eval /= count
        loss_eval_div /= count
        loss_eval_recon /= count

        for key in avg_individual_corrs:
            if count > 0: # Evita división por cero si el loader está vacío
                avg_individual_corrs[key] = -avg_individual_corrs[key] / count
    
        if use_multi_target:
            return loss_eval, loss_eval_div, loss_eval_recon, avg_individual_corrs, mean_distance_to_mean, var_latent_mean
        else:
            # --- CAMBIO 2: DEVUELVE EL VALOR DE ADAS13 ---
            return loss_eval, loss_eval_div, loss_eval_recon, avg_individual_corrs['ADAS13'], mean_distance_to_mean, var_latent_mean

#_________________________________

#           TESTING
#__________________________________



def test(test_loader, model, results_folder):
    """
    This function performs testing on the test dataset.
    Set model to .eval() and torch.no_grad() to ensure model
    does not change. Encode and decode test images using trained model.
    Compute loss between real images and reconstructed data to check validity
    of the model.
    """
    model.eval()

    beta_corr_dict = {
        target: beta_corr_weights[target]
        for target in correlation_targets
    }
        
    loss_test = 0.0
    count = 0
    
    z_list = []
    z_mean_list = []
    z_logvar_list = []
    
    feature_test_dict = {}
    
    print("Starting testing loop...")
    with torch.no_grad():
        for x_img, feature_dict in test_loader:
            
            x_img = x_img.to(device)
            
            #Create the dict with targets
            targets = {
                'ADAS13': feature_dict['ADAS13'].to(device),
                'CDRSB': feature_dict['CDRSB'].to(device)
            }
            
            recon_batch, z_sampled, z_mean, z_logvar = model(x_img)
            
            if len(recon_batch.shape) == 5: # Remove channel dim
                recon_batch = recon_batch.squeeze(1) # [BATCH, DIMS] 
        
            if len(x_img.shape) == 5: # Remove channel dim
                x_img = x_img.squeeze(1)

            # Loss is used only to compute the total final testing loss.
            loss_batch, _, _, _, _ = model.loss(x_img, recon_batch, z_sampled, z_mean, z_logvar, beta_corr_dict, targets)
            
            loss_test += loss_batch.item()
            count += 1
            
            z_list.append(z_sampled.detach().cpu())
            z_mean_list.append(z_mean.detach().cpu())
            z_logvar_list.append(z_logvar.detach().cpu())
            
            if not feature_test_dict:
                feature_test_dict = {key: [] for key in feature_dict.keys()}
                
            for key, val in feature_dict.items():
                if isinstance(val, torch.Tensor):
                    feature_test_dict[key].append(val.cpu())
                else:
                    feature_test_dict[key].append(val)
                    
                        
        print(f"Test finished. Avg loss: {loss_test / count:.4f}")
        
        # ================
        # Post-Processing
        # ================
                
        #We concatenate the lists of latents t9o have one single tensor
        z_test_tensor = torch.cat(z_list, dim = 0)
        z_mean_tensor = torch.cat(z_mean_list, dim = 0)
        z_logvar_tensor = torch.cat(z_logvar_list, dim = 0)
        
        #_____________________________
        # Write results in csv files
        #_____________________________
        
        # Save latent space of testing subjects
        # We use the concatenated tensors instead of lists
        z_array = z_test_tensor.numpy()
        df_z = pd.DataFrame(z_array)
        df_z.to_csv(os.path.join(results_folder, 'z_test.csv'), index = False)  
        
        # Convert z_mean_list to dataframe
        z_mean_array = z_mean_tensor.numpy()
        df_z_mean = pd.DataFrame(z_mean_array)
        df_z_mean.to_csv(os.path.join(results_folder, 'z_mean_test.csv'), index = False)
        
        z_logvar_array = z_logvar_tensor.numpy()
        df_z_logvar = pd.DataFrame(z_logvar_array)
        df_z_logvar.to_csv(os.path.join(results_folder, 'z_logvar_test.csv'), index = False)   
        
        # Now we process the dictionary of features.
        # To adjust the format into a single list instead of list of lists (flatten)      
        ptid_list = [item for sublist in feature_test_dict['PTID'] for item in sublist]
        viscode_list = [item for sublist in feature_test_dict['VISCODE'] for item in sublist]
        
        # Now we process the rest of numerical features           
        for key, value_list in feature_test_dict.items():
            
            # feature_test_dict can contain lists of tensors, numbers, lists of lists, etc.
            # We want 1 single tensor for feature [ADAS13: 3.2, 0, 12.2, ..., DX: 0, 1, 1, 2, 0, ..., etc.]
            # We have to process the data from each feature to read it
            
            if key in['PTID', 'VISCODE']: # Ignore the keys PTID and VISCODE
                continue
            
            """
            This block takes into account different possible formats. 
            value_list = ["string1", "string2", ...]. In this case --> isinstance(value_list[0], str) = true,
            value_list = [["string1", "string2"], ["string3", "string4"], ...]. In this case --> isinstance(value_list[0], list) and isinstance(value_list[0][0], str) = true,
            """
            if isinstance(value_list[0], str) or isinstance(value_list[0], list) and isinstance(value_list[0][0], str):
                continue # Ignore features with strings, it processes only numerical data

            """
            This part converts structures like [[1,2], 3, [4,5]] into [1,2,3,4,5]
            by flattening.
            """
            flattened_list = []
            for v in value_list:
                if isinstance(v, list):
                    flattened_list.extend(v)
                else:
                    flattened_list.append(v)

            if isinstance(flattened_list[0], torch.Tensor):
                feature_test_dict[key] = torch.cat(flattened_list, dim = 0)
            else:
                feature_test_dict[key] = torch.tensor(flattened_list)

        if 'ADAS13' in feature_test_dict and isinstance(feature_test_dict['ADAS13'], torch.Tensor):
            adas13_arr = feature_test_dict['ADAS13'].numpy()
        else:
            adas13_arr = np.zeros(z_array.shape[0])
            
        if 'AGE' in feature_test_dict and isinstance(feature_test_dict['ADAS13'], torch.Tensor):
            age_arr = feature_test_dict['AGE'].numpy()
        else:
            age_arr = np.zeros(z_array.shape[0])

            
        latent_dim = z_array.shape[1]
        
        """
        This creates a dataframe where the first two columns are PTID and VISCODE,
        while the rest are the latent variables.
            PTID         VISCODE     z_0    z_1   z_2   z_3 ...
        0  011_S_0002       bl       1.1   -0.3  -2.1   3.0 
        1  011_S_0003       bl       -1.6   1.3   2.7   0.32
        2  011_s_0003       m06      2.1    0.41  -1.1  0.2
        ...
        
        This allows to track each latent space in the longitudinal model.
        """
        df_z_mean = pd.DataFrame(z_mean_array, columns = [f'z_{i}' for i in range(latent_dim)])
        df_z = pd.DataFrame(z_array, columns = [f'z_{i}' for i in range(latent_dim)])
        
        # Insert PTID and VISCODE at the beggining of the dataframe
        
        df_z_mean['ADAS13'] = adas13_arr
        df_z_mean['AGE'] = age_arr
        df_z_mean.insert(0, 'VISCODE', viscode_list)
        df_z_mean.insert(0, 'PTID', ptid_list)

        df_z['ADAS13'] = adas13_arr #ADD the ADAS13 array to the dataframe 
        df_z['AGE'] = age_arr
        df_z.insert(0, 'VISCODE', viscode_list)
        df_z.insert(0, 'PTID', ptid_list)
        
        if not os.path.exists("longitudinal_results"):
            os.makedirs("longitudinal_results")
   
        # Save CSV file of latent space + IDs
        df_z_mean.to_csv(os.path.join("longitudinal_results", 'z_mean_test_with_IDs.csv'), index = False )
        df_z.to_csv(os.path.join(results_folder, 'z_test_with_IDs.csv'), index = False )
            
        # Save a CSV file [z_mean, feature]
        for feature in feature_test_dict.keys():
            if feature not in ['PTID', 'VISCODE']:
                df_test = pd.DataFrame(z_array, columns=[f'z_{i}' for i in range(latent_dim)])
                df_test[feature] = feature_test_dict[feature].numpy()
                df_test.to_csv(os.path.join(results_folder, f'z_mean_{feature}_TEST.csv'), index=False)

    return z_test_tensor, z_mean_tensor, z_logvar_tensor, feature_test_dict
            
#___________________________________________________________________________________

                # MAIN FUNCTION FOR USE FROM RUN.PY FOLDER #
                
def main(config_path = 'config.yaml', results_folder=None, path_ADNIMERGE=None):
    """Main function to execute the VAE model.
        
    Args:
        -config_path: path to config.yaml
        -results_folder: where to save results and models
        -data_folder: root folder containing data (ADNIMERGE.csv, etc.)
            
    Returns:
        - dict: containing model, results, and metrics"""
    
    print("\n" + "="*60)
    print("VAE MODEL TRAINING PIPELINE")
    print("="*60)
    
    os.makedirs(results_folder, exist_ok=True)
    
    # -------------------------------------
    # Step 1: Load configuration
    # -------------------------------------
    
    print(f"\n[1/6] Loading configuration from {config_path}...")
    config = load_config(config_path)
    shutil.copy(config_path, results_folder)
    
    # Extract hyperparameters
    div_criteria = config['model']['divergence']
    latent = config['model']['latent']
    batch_size = config['model']['batch_size']
    device_name = config['experiment']['device']
    recon_function = config['model']['reconstruction']
    beta = config['model']['loss']['beta']
    optimizer_model = config['model']['optimizer']
    learning_rate = config['model']['lr']
    epochs = config['model']['epochs']
    architecture = config['model']['architecture']
    correlation_targets = config['model']['correlation_targets']
    beta_corr_weights = config['model']['loss']['beta_corr_weights']
    
    print(f"   ✓ Divergence: {div_criteria}")
    print(f"   ✓ Reconstruction: {recon_function}")
    print(f"   ✓ Latent dimensions: {latent}")
    print(f"   ✓ Architecture: {architecture}")
    
    #Now, we set the data paths (which are indicated by the run.py)
    
    features = [
        (config_key, score_name) 
        for config_key, score_name in config['results'].items()
        if config_key.startswith('feature_')
    ]
    
    # -------------------------------------
    # Step 2: Setup device and logging
    # -------------------------------------
    
    print(f"\n[2/6] Setting up device and logging...")
    if sys.stdout.isatty():
        init(autoreset=True)
    else:
        init(autoreset=True, strip=True)
        
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"   ✓ Using device: {device}")
    
    model_save_path = os.path.join(results_folder, 'vae_model.pth')
    
    # -------------------------------------
    # Step 3: Load data
    # -------------------------------------
    print(f"\n[3/6] Loading data...")
    
    try:
        df_ADNIMERGE = pd.read_csv(path_ADNIMERGE)
        print(f"    ✓ Loaded ADNIMERGE: {len(df_ADNIMERGE)} records")
        
        train_loader, eval_loader, test_loader = get_dataloader(
            batch_size,
            shuffle=True,
            num_workers=16,
            random_seed=42
        )
        verify_subject_separation(train_loader, eval_loader, test_loader)
        print(f"   ✓ Dataloaders created successfully")
    
    except Exception as e:
        print(f"    ❌ [ERROR] Data could not be loaded")
        
     # -------------------------------------
    # Step 4: Initialize and train model
    # -------------------------------------
    print(f"\n[4/6] Training model...")
    print("="*60)
    
    try:
        #We initializate the model
        model=model_VAE()
        model=model.to(device)
        
        #We setup the optimizer
        if optimizer_model == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_model == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_model}")
        
        # Early stopping
        early_stopping = EarlyStopping(results_folder=results_folder,
                                       patience=100)
        
        # Train
        model = train(model, train_loader, eval_loader, early_stopping, results_folder)
        print("   ✓ Training completed successfully")
        
    except Exception as e:
        print(f"   ❌ ERROR during training: {str(e)}")
        raise 
    
    # -------------------------------------
    # Step 5: Load best model and test
    # -------------------------------------
    print(f"\n[5/6] Testing model...")
    
    try:
        #We load the best checkpoint of the model
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✓ Loaded best model from epoch {checkpoint['epoch']}")
        
        # Test
        z_test_list, z_mean_list, z_logvar_list, feature_test_dict_list = test(
            test_loader, 
            model,
            results_folder
        )
        print("   ✓ Testing completed successfully")
        
    except Exception as e:
        print(f"   ❌ ERROR during testing: {str(e)}")
        raise
    
    
    # -------------------------------------
    # Step 6: Generate results and plots
    # -------------------------------------
    print(f"\n[6/6] Generating results and visualizations...")

    results = {
        'model': model,
        'checkpoint': checkpoint,
        'z_test': z_test_list,
        'z_mean': z_mean_list,
        'z_logvar': z_logvar_list,
        'features': feature_test_dict_list,
        'config': config
    }
    
    try:
        # Plot latent vs features
        print("   → Plotting latent space vs features...")
        for config_key, score_name in features:
            plot_latentx_vs_feature(z_mean_list, z_test_list, feature_test_dict_list, score_name, results_folder)
        print("   ✓ Feature plots completed")
        
        # Get training latents for SVR
        print("   → Extracting training latents for SVR...")
        z_train_list = []
        feature_dicts_train = {key: [] for key in next(iter(train_loader))[1].keys()}
        
        model.eval()
        with torch.no_grad():
            for x_img, feature_dict in train_loader:
                x_img = x_img.to(device)
                
                for key, value in feature_dict.items():
                    feature_dicts_train[key].append(value)
                
                _, _, z_mean_train_batch, _ = model(x_img)
                z_train_list.extend(z_mean_train_batch)
            
            # Process feature dictionaries
            for key, value_list in feature_dicts_train.items():
                flattened_list = []
                for v in value_list:
                    if isinstance(v, list):
                        flattened_list.extend(v)
                    else:
                        flattened_list.append(v)
                
                cleaned_list = []
                for v in flattened_list:
                    if isinstance(v, torch.Tensor):
                        cleaned_list.append(v)
                    elif isinstance(v, (int, float, list, np.ndarray)):
                        cleaned_list.append(torch.tensor(v))
                    elif isinstance(v, str):
                        continue
                    else:
                        raise TypeError(f"Unexpected type in key '{key}': {type(v)}")
                
                if cleaned_list:
                    feature_dicts_train[key] = torch.cat(cleaned_list, dim=0)
        
        results['z_train'] = z_train_list
        results['features_train'] = feature_dicts_train
    
        # SVR analysis
        print("   → Running SVR analysis...")
        for config_key, score_name in features:
            SVR_feature(z_test_list, z_train_list, feature_dicts_train, 
                       feature_test_dict_list, score_name, results_folder)
        print("   ✓ SVR analysis completed")
        
        # Classification tasks
        print("   → Running classification tasks...")
        class_AD_vs_CN(z_test_list, z_train_list, feature_test_dict_list, feature_dicts_train, results_folder)
        class_AD_vs_CN_ADAS(feature_test_dict_list, feature_dicts_train, results_folder)
        class_AD_vs_CN_First_Latent(z_test_list, z_train_list, feature_test_dict_list, feature_dicts_train, results_folder)
        print("   ✓ Classification completed")
        
    except Exception as e:
        print(f"   ⚠ WARNING: Error during visualization/analysis: {str(e)}")
        print("   Model training completed successfully, but some plots may be missing")
    
    # -------------------------------------
    # Summary
    # -------------------------------------
    print("\n" + "="*60)
    print("✓ VAE PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Results saved to: {results_folder}")
    print(f"Model saved to: {model_save_path}")
    print(f"Best validation loss: {checkpoint.get('loss_eval', 'N/A')}")
    print(f"Training epochs: {checkpoint.get('epoch', 'N/A')}")
    print("="*60 + "\n")
    
    return results
    
    
    
#______________________________________________________________________________#
                    # If executed directly from file #
                    
if __name__ == '__main__':
    default_results = os.path.join('RESULTS', 'beta_vae_results')
    main(
        config_path='config.yaml',
        results_folder=default_results,
        path_ADNIMERGE=config['loader']['load_ADNIMERGE'],
    )
    
    
    
    
    

#___________________________________________________________________________________




# if __name__ == '__main__':
    
#     if sys.stdout.isatty():
#        init(autoreset=True)
#     else:
#         init(autoreset=True, strip=True) 
        

    
#     model_save_path = f'{results_folder}/vae_model.pth'
#     if not os.path.exists(results_folder):
#         os.makedirs(results_folder)
        

#     print(f'Using divergence {div_criteria}')
#     print(f'Using reconstruction {recon_function}')
     
#     device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    
#     print(f"Using device: {device}")
    
#     df_ADNIMERGE = pd.read_csv(path_ADNIMERGE)
    
    
#     #____________ LOADER OF IMAGES AND FEATURES _______________
    
#     train_loader, eval_loader, test_loader = get_dataloader(batch_size, shuffle = True, num_workers = 16, random_seed = 42)
#     verify_subject_separation(train_loader, eval_loader, test_loader)
    
#     print(Fore.MAGENTA + f'Loading of images terminated.')
    
#     #__________________________________________________________
    
#     #LOAD MODEL AND TRAIN
#     model = model_VAE()
    
#     early_stopping = EarlyStopping(patience=100, save_path=f'{results_folder}/vae_model.pth') # Stop after X epochs if there is no improvement
    
#     print('____________________________________________')
#     print(Fore.RED + 'Start of training')
#     model = train(model, train_loader, eval_loader)  
#     print(Fore.RED + 'Training terminated')
#     print('____________________________________________')
    
#     print('Loading best saved model')
#     checkpoint = torch.load(model_save_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     #___________________________TESTING_________________________
    
#     z_test_list, z_mean_list, z_logvar_list, feature_test_dict_list = test(test_loader, model) #IF TRAIN_SET IS HERE, CHANGE FOR TEST_SET.
#     print(Fore.CYAN + 'Test terminated')
    
#     #___________________________________________________________
    
#     print(Fore.YELLOW + 'Starting plot of results')
    


#     print(Fore.BLUE + 'Starting LATENTX_VS_FEATURE')
    
#     #PLOT LATENTX VS VALUE OF FEATURE
#     for config_key, score_name in features:
#         plot_latentx_vs_feature(z_mean_list, z_test_list, feature_test_dict_list, score_name)
#         print(f'Plot of latentx vs {score_name} finalized')


#     print(Fore.BLUE + 'Starting SVR')
    
#     # GET Z_TRAIN TO FEED SVR
    
#     if architecture == 'VAE':
#         z_train_list = []
#         feature_dicts_train = {key: [] for key in next(iter(train_loader))[1].keys()}
    
#         with torch.no_grad():
#             for x_img, feature_dict in train_loader:
#                 x_img = x_img.to(device)
            
#                 for key, value in feature_dict.items():
#                     feature_dicts_train[key].append(value)
            
#                 _, _, z_mean_train_batch, _ = model(x_img)
            
#                 z_train_list.extend(z_mean_train_batch)
        
#             for key, value_list in feature_dicts_train.items():
#                 # Aplanar listas internas
#                 flattened_list = []
#                 for v in value_list:
#                     if isinstance(v, list):
#                         flattened_list.extend(v)
#                     else:
#                         flattened_list.append(v)
                
#                 # Convertir solo elementos numéricos a tensor y saltar strings
#                 cleaned_list = []
#                 for v in flattened_list:
#                     if isinstance(v, torch.Tensor):
#                         cleaned_list.append(v)
#                     elif isinstance(v, (int, float, list, np.ndarray)):
#                         cleaned_list.append(torch.tensor(v))
#                     elif isinstance(v, str):
#                         # Ignorar strings o imprimir advertencia si quieres
#                         #print(f"Ignorando string en key '{key}': {v}")
#                         continue
#                     else:
#                         raise TypeError(f"Tipo inesperado en key '{key}': {type(v)}")
                
#                 if cleaned_list:  # Solo hacer cat si la lista no está vacía
#                     feature_dicts_train[key] = torch.cat(cleaned_list, dim=0)
#                 else:
#                     print(f"No hay tensores para concatenar en la key '{key}'")

                
    
#     #COMPUTE SVR OF FEATURES
#     for config_key, score_name in features:
#         SVR_feature(z_test_list, z_train_list, feature_dicts_train, feature_test_dict_list, score_name)
#         print(f'SVR of {score_name} finalized')
    
    
#     class_AD_vs_CN(z_test_list, z_train_list, feature_test_dict_list, feature_dicts_train)
#     class_AD_vs_CN_ADAS(feature_test_dict_list, feature_dicts_train)
#     class_AD_vs_CN_First_Latent(z_test_list, z_train_list, feature_test_dict_list, feature_dicts_train)
    
   