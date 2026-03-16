from .load_database import merge_id_ses_to_ADNIMERGE, merge_lists
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statsmodels.api as sm
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from config import load_config
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import linregress
from pathlib import Path

config_file = 'config.yaml'
config = load_config(config_file)

#LOAD HYPERPARAMETERS FROM CONFIG FILE

#We load the saving folders relative to root folder
# root_folder = config["root"]
# root_folder = Path(root_folder) #Converts string to path 
# results_folder = root_folder / config["folders"]["results"]

# pca_folder = results_folder / 'pca_folder'
# filtered_lists_svr_folder = results_folder / 'filtered_lists_svr_folder'

# # For latentx vs feature plots
# path_dir_folder = results_folder / 'path_dir_folder'

# if not os.path.exists(pca_folder):
#     os.makedirs(pca_folder)

# if not os.path.exists(results_folder):
#     os.makedirs(results_folder)
        
# if not os.path.exists(filtered_lists_svr_folder):
#     os.makedirs(filtered_lists_svr_folder)
    
debugging = config['experiment']['debugging']
latent_dim = config['model']['latent']
device = config['experiment']['device'] # Device to perform computation


#__________________________________________________________________

#                   FUNCTIONS FOR FEATURES
#__________________________________________________________________


def SVR_feature(z_list, z_train_list, feature_dicts_train, feature_dicts_test, feature, results_folder):
    """
    Function for training a Support Vector Regression model to predict a feature of interest using the latent space.
        Args:
            - z_list: List of latent space of every test subject.
            - feature_dicts: Dictionary of features of every test subject.
            feature_dicts has the following structure:
            {
                'feature1': [value1, value2, ...],
                'feature2': [value1, value2, ...],
                ...
            }
            Here, I extract the list of values for the feature of interest.
            (Which is the same order as z_list).
            -feature: feature of interest (ADAS13, VENTRICLES, MIDTEMP, ETC)
    """
    
    mse_r2_folder = os.path.join(results_folder, 'mse_r2_folder')
    os.makedirs(mse_r2_folder, exist_ok=True)

    # Read feature list and convert to numpy for test and train
    feature_list_test = np.array(feature_dicts_test[f'{feature}'])
    mask_test = ~np.isnan(feature_list_test)
    
    feature_list_train = np.array(feature_dicts_train[f'{feature}'])
    mask_train = ~np.isnan(feature_list_train)
    
    # Convert latent space from tensor to numpy 
    clean_z_list = [z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z for z in z_list]
    clean_z_train_list = [z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z for z in z_train_list]
    
    # Apply nan mask to latent space and feature list to both test and train
    clean_z_list_test = np.array(clean_z_list)[mask_test]
    clean_z_list_train = np.array(clean_z_train_list)[mask_train]
    
    clean_feature_list_test = (feature_list_test)[mask_test] 
    clean_feature_list_train = (feature_list_train)[mask_train]    
    
    ### SELECT IF YOU WANT SVR WITH ENTIRE LATENT SPACE, OR WITH JUST FIRST DIMENSION ###
       
       
    ##### ===== FULL LATENT SPACE ====== ####
    X_test = np.array([z for z in clean_z_list_test], dtype = np.float64) # Convert to 2D array
    X_train= np.array([z for z in clean_z_list_train], dtype = np.float64) # Convert to 2D array
    result_folder = os.path.join(mse_r2_folder, 'ALL_latents')
    
    
    ##### ===== FIRST LATENT DIMENSION ===== ####
    # X_test = np.array([[z[0]] for z in clean_z_list_test], dtype=np.float64)
    # X_train = np.array([[z[0]] for z in clean_z_list_train], dtype=np.float64)
    # result_folder = os.path.join(mse_r2_folder, 'FIRST_latent_only')
    
    # TARGET VARIABLES (ADAS, HIPPOCAMPUS, ETC)
    y_train = np.array(clean_feature_list_train, dtype = np.float64) # Convert to 1D array
    y_test = np.array(clean_feature_list_test, dtype = np.float64) # Convert to 1D array


    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError(f'No valid training data for {feature}, skipping  SVR fitting')
    
    # Train SVR model
    model = make_pipeline(StandardScaler(), LinearSVR(dual = 'auto'))
    
    regr = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())

    
    #Train model
    regr.fit(X_train, y_train)
    
    #Predict for test set
    y_pred = regr.predict(X_test)
    
    #Evaluate model with MSE
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean squared error of test and prediction for {feature}: {mse}')
    print(f'R2 score of test and prediction for {feature}S: {r2}')
    
    svr_model = regr.regressor_.named_steps['linearsvr']
    
    print(f'Coefficients of LinearSVR model for {feature}: {svr_model.coef_}')
    
    with open(os.path.join(result_folder, f'MSE_R2_SVR_{feature}.txt'), 'w') as file:
        file.write(f'{mse}, {r2}\n')
            
    with open(os.path.join(result_folder, f'SVR_COEFFS_{feature}.txt'), 'w') as file:
        file.write(f'{svr_model.coef_}\n')
        
        



def plot_latentx_vs_feature(z_mean_list, z_list, feature_dicts, feature, results_folder):
    """
    Function for plotting latent_dim x against latent_dim.
        Args:
            - z_list: List of latent space of every test subject.
            - feature_dicts: Dictionary of features of every test subject.
            feature_dicts has the following structure:
            {
                'feature1': [value1, value2, ...],
                'feature2': [value1, value2, ...],
                ...
            }
            Here, I extract the list of values for the feature of interest.
            (Which is the same order as z_list).
            - feature of interest
    """
    path_dir_folder = os.path.join(results_folder, 'path_dir_folder')
            
    if not os.path.exists(path_dir_folder):
        os.makedirs(path_dir_folder)
    
    sub_folder = os.path.join(results_folder, f"scatter_images_latx_VS_{feature}")
    
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    
    latents_plot = np.arange(latent_dim)
    
    # Convert dictionary to list for feature of interest
    feature_list = np.array(feature_dicts[f'{feature}'])
    dx_list = np.array(feature_dicts['DX']) # GET THE
    
    z_list = [z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z for z in z_list]
    z_mean_list = [z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z for z in z_mean_list]
    
    mask = ~np.isnan(feature_list)
    clean_z_list = np.array(z_list)[mask]
    clean_z_mean_list = np.array(z_mean_list)[mask]
    clean_feature_list = (feature_list)[mask]   
    clean_dx_list = dx_list[mask] 
    
    # FOR SAVING THE CORRELATION VALUE OF THE GRAPHS
    coef_file = os.path.join(sub_folder, f'regression_coefficients_{feature}.text')
    with open(coef_file, 'w') as f:
        f.write('LatentDim,Coef_regresion,Coef_correlation,p_val\n')
    

        for latx in latents_plot:
            
            #PLOT ZMNEAN VS FEATURE
            
            dim_x_mean = [item[latx] for item in clean_z_mean_list] #GET DIM X OF LATENT SPACE
            
            #Linear regression with scipy
            slope, intercept, r_value, p_value, std_err = linregress(dim_x_mean, clean_feature_list)
            #print(f'Latent dim {latx}: coef regresión = {slope:.4f}, r = {r_value:.4f}, p = {p_value:.4e}')
            f.write(f'{latx},{slope},{r_value},{p_value}\n')
                    
            fig, ax = plt.subplots(figsize=(8, 6))

            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=0, vmax=2)
            diagnosis_labels = ['CN', 'MCI', 'AD']
            markers = ['o', 's', 'D']

            for i, marker in enumerate(markers):
                mask = (clean_dx_list == i)
                color = cmap(norm(i))
                ax.scatter(np.array(dim_x_mean)[mask], np.array(clean_feature_list)[mask], 
                        color=color, marker=marker, label=diagnosis_labels[i], alpha=0.7)

            # Línea de regresión (solo línea)
            slope, intercept, r_value, p_value, std_err = linregress(dim_x_mean, clean_feature_list)
            x_vals = np.linspace(np.min(dim_x_mean), np.max(dim_x_mean), 100)
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, color='black', linestyle='--', label='Regresión')

            ax.set_xlabel(f'Latent dim {latx}')
            ax.set_ylabel(f'{feature}')

            # Leyenda solo con colores y formas de los diagnósticos
            ax.legend(title='Diagnosis')

            plt.savefig(f'{sub_folder}/scatter_plot_z_mean_dim{latent_dim}_lat{latx}.png', format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        
    return None


def class_AD_vs_CN(z_list_test, z_list_train, diagnosis_test, diagnosis_train, results_folder):
    
    y_test = np.array(diagnosis_test['DX'])
    y_train = np.array(diagnosis_train['DX'])

    mask_test = ~np.isnan(y_test)
    mask_train = ~np.isnan(y_train)

    # Filtrar solo CN (0) y AD/Dementia (2)
    keep_test = np.isin(y_test, [0, 2])
    keep_train = np.isin(y_train, [0, 2])
    
    combined_mask_test = mask_test & keep_test
    combined_mask_train = mask_train & keep_train

    z_list_test_cpu = [z.detach().cpu().numpy() for z in z_list_test]
    z_list_test_array = np.vstack(z_list_test_cpu)[combined_mask_test]
    z_list_train_cpu = [z.detach().cpu().numpy() for z in z_list_train]
    z_list_train_array = np.vstack(z_list_train_cpu)[combined_mask_train]

    y_test = y_test[combined_mask_test]
    y_train = y_train[combined_mask_train]

    # Optional: reassign labels to 0 (CN) and 1 (AD)
    y_test = (y_test == 2).astype(int)
    y_train = (y_train == 2).astype(int)

    df_test = pd.DataFrame(z_list_test_array, columns=[f"z{i}" for i in range(latent_dim)])
    df_test['DX'] = y_test

    print(f'Class distribution in AD vs CN classification for test set: {pd.Series(y_test).value_counts()}')
    print(f'Class distribution in AD vs CN classification for train set: {pd.Series(y_train).value_counts()}')

    ##### === BOXPLOT TO VISUALIZE THE CONTRIBUTION OF FIRST LATENT
    ##### AND THE REST OF THE LATENTS === #####
    
    for i in range(latent_dim):
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='DX', y=f'z{i}', data=df_test, palette='Set2', linewidth = 0.8, width = 0.4)
        plt.title(f'Boxplot of latent variable z_{i} (AD vs CN)')
        plt.xlabel('Diagnosis (DX)')
        plt.ylabel(f'Value of $z_{i}$')
        plt.xticks([0, 1], ['CN', 'AD'])
        plt.tight_layout()
        plt.savefig(f'{results_folder}/boxplot_ADvsCN_latent_{i}.png')
        plt.close()
        
        #COMBINED BOXPLOT
    df_test = pd.DataFrame(
        z_list_test_array,
        columns=[f"$z_{{{i+1}}}$" for i in range(latent_dim)]
    )
    df_test['DX'] = ['CN' if dx == 0 else 'AD' for dx in y_test]  # Etiquetas como texto

    df_melted = df_test.melt(id_vars='DX', var_name='latent_variable', value_name='value')

    plt.figure(figsize=(14, 6))
    sns.boxplot(x='latent_variable', y='value', hue='DX', data=df_melted,
                palette='Set2', width=0.4, linewidth=0.7)

    plt.xlabel('Latent variable')
    plt.ylabel('Value')
    plt.title('Boxplot of all latent variables (AD vs CN)')
    plt.legend(title='Diagnosis')
    plt.tight_layout()
    plt.savefig(f'{results_folder}/combined_boxplot_ADvsCN.png')
    plt.close()

    clf = LogisticRegression(max_iter = 1000)
    clf.fit(z_list_train_array, y_train)
    y_pred = clf.predict(z_list_test_array)
    
    df_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    df_results.to_csv(f'{results_folder}/y_true_vs_y_pred_ADvsCN.csv', index = False)

    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=['CN', 'AD'])

    with open(f'{results_folder}/classification_report_ADvsCN.txt', 'w') as f:
        f.write(report)    
        

    return None


def class_AD_vs_CN_First_Latent(z_list_test, z_list_train, diagnosis_test, diagnosis_train, results_folder):
    
    y_test = np.array(diagnosis_test['DX'])
    y_train = np.array(diagnosis_train['DX'])

    mask_test = ~np.isnan(y_test)
    mask_train = ~np.isnan(y_train)

    # Filtrar solo CN (0) y AD/Dementia (2)
    keep_test = np.isin(y_test, [0, 2])
    keep_train = np.isin(y_train, [0, 2])
    
    combined_mask_test = mask_test & keep_test
    combined_mask_train = mask_train & keep_train

    z_list_test_cpu = [z.detach().cpu().numpy() for z in z_list_test]
    z_list_test_array = np.vstack(z_list_test_cpu)[combined_mask_test]
    z_list_train_cpu = [z.detach().cpu().numpy() for z in z_list_train]
    z_list_train_array = np.vstack(z_list_train_cpu)[combined_mask_train]
    
    ## CHOOSE FIRST LATENT VARIABLE TO CLASSIFY
    
    z_list_train_array_var0 = z_list_train_array[:, 0].reshape(-1, 1)
    z_list_test_array_var0 = z_list_test_array[:, 0].reshape(-1, 1)

    y_test = y_test[combined_mask_test]
    y_train = y_train[combined_mask_train]

    # Optional: reassign labels to 0 (CN) and 1 (AD)
    y_test = (y_test == 2).astype(int)
    y_train = (y_train == 2).astype(int)

    df_test = pd.DataFrame(z_list_test_array, columns=[f"z{i}" for i in range(latent_dim)])
    df_test['DX'] = y_test

    print(f'Class distribution in AD vs CN classification for test set: {pd.Series(y_test).value_counts()}')
    print(f'Class distribution in AD vs CN classification for train set: {pd.Series(y_train).value_counts()}')

    plt.figure(figsize=(6,4))
    sns.boxplot(x='DX', y= 'z0', data=df_test, palette='Set2')
    plt.title('Boxplot using only z_0 for diagnosis classification (AD vs CN)')
    plt.xlabel('Diagnosis (DX)')
    plt.ylabel(r'Value of $z_0$')
    plt.xticks([0,1], ['CN', 'AD'])
    plt.savefig(f'{results_folder}/boxplot_ADvsCN_class_one_var')
    plt.close()

    clf = LogisticRegression(max_iter = 1000)
    clf.fit(z_list_train_array_var0, y_train)
    y_pred = clf.predict(z_list_test_array_var0)
    
    df_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    df_results.to_csv(f'{results_folder}/y_true_vs_y_pred_ADvsCN_first_Latent.csv', index = False)

    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=['CN', 'AD'])

    with open(f'{results_folder}/classification_report_ADvsCN_first_Latent.txt', 'w') as f:
        f.write(report)    
        

    return None


def class_AD_vs_CN_ADAS(diagnosis_test, diagnosis_train, results_folder):
    # diagnosis_ is expected to be a dictionary of the test subjects
    
    # Get diagnosis of subjects
    y_test = np.array(diagnosis_test['DX']) # DX = CN, DX = MCI, DC = Dementia
    y_train = np.array(diagnosis_train['DX'])
    # For AD_vs_CN using ADAS
    adas_test = np.array(diagnosis_test['ADAS13'])
    adas_train = np.array(diagnosis_train['ADAS13'])

    # Remove NaNs and filder for CN and Dementia. Use 0 for CN and 2 for dementia
    mask_test = ~np.isnan(y_test) & ~np.isnan(adas_test) & np.isin(y_test, [0, 2])
    mask_train = ~np.isnan(y_train) & ~np.isnan(adas_train) & np.isin(y_train, [0, 2])

    X_test = adas_test[mask_test].reshape(-1, 1)
    X_train = adas_train[mask_train].reshape(-1, 1)
    y_test = (y_test[mask_test] == 2).astype(int)   # AD = 1, CN = 0
    y_train = (y_train[mask_train] == 2).astype(int)

    print(f'Class distribution in test set (ADAS): {pd.Series(y_test).value_counts()}')
    print(f'Class distribution in train set (ADAS): {pd.Series(y_train).value_counts()}')

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    df_boxplot = pd.DataFrame({
        'ADAS13': adas_test[mask_test],
        'Diagnosis': np.where(y_test == 0, 'CN', 'AD')
    })
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Diagnosis', y='ADAS13', data=df_boxplot, palette='Set2', width=0.4, linewidth=0.7)
    plt.title('Boxplot of ADAS13 for diagnosis classification (AD vs CN)')
    plt.ylabel('ADAS13 Score')
    plt.xlabel('Diagnosis')
    plt.savefig(f'{results_folder}/boxplot_ADvsCN_class_ADAS')
    plt.close()

    
    # SAVE Y_PRED AND Y_TRUE TO COMPUTE CLASSIFICATION METRICS
    df_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    df_results.to_csv(f'{results_folder}/y_true_vs_y_pred_ADvsCN_ADAS.csv', index = False)

    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, target_names=['CN', 'AD'])

    with open(f'{results_folder}/classification_report_ADvsCN_ADAS.txt', 'w') as f:
        f.write(report)

    return None






def tsne_latent(z, feature_dict, feature, results_folder):
    """
    t-SNE: non-linear dimensional reduction technique.
    """
    
    sub_folder = os.path.join(results_folder, 'tSNE')
 
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
        
    feature_list = np.array(feature_dict[f'{feature}'].tolist())

    clean_z_list, clean_feature_list = zip(*[
    (z, f) for z, f in zip(z, feature_list)
    if not (np.isnan(f) or f == -1)
    ])
    
    clean_z_array = np.array([z.detach().cpu().numpy() for z in clean_z_list]) #Move to CPU and convert to numpy
    clean_feature_array= np.array(clean_feature_list)
    
    scaler = StandardScaler() #To normalize latent space
    clean_z_array = scaler.fit_transform(clean_z_array)
    
    # Larger perplexity captures global structure. Lower perplexity captures local structure
    tsne = TSNE(n_components = 2, random_state = 42, perplexity = 30, n_iter = 1000)
    latent_2d = tsne.fit_transform(clean_z_array)
    
    plt.figure(figsize = (8,6))
    scatter = plt.scatter(latent_2d[:,0], latent_2d[:,1], c = clean_feature_array, cmap = 'viridis', alpha = 0.7)
    plt.colorbar(scatter, label = f'{feature}')
    plt.title('t-SNE visualization of latent space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    plt.savefig(f'{sub_folder}/tsne_visualization{feature}.png', dpi = 300, bbox_inches = 'tight')
    plt.close()





def plot_latent_space_feature(z_list, id_ses_list, df_ADNIMERGE, path_test, config, feature, results_folder):
    """
    Function for plotting latent_dim x against latent_dim for test_set.
    Each point in the scatterplot is colorcoded by its feature score.
        Args:
        
            -z_list: List of dimensions (length(test_set), lat_space). Contains
            the latent space of every test_set subject (or whatever we feed the function)
            
            -test_id_ses: List of tuples (ID, SES, feature) of each test_set subject. The order
            of z_list matches the order of test_id_ses (they are sorted together)
            
            -df_ADNIMERGE: ADNIMERGE dataframe containing the information of every subject of
            the study (More subjects than our database).
            
            -config: configuration file with hyprparameters. Used to obtain latent_dim.
    """
    
    sub_folder = os.path.join(results_folder, f"scatter_images_{feature}")
    
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    
    latent_dim = config['model']['latent']
    latents_plot = np.arange(latent_dim)
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, feature)
    df_ID_SES_feature = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_feature = df_ID_SES_feature.set_index(['PTID', 'VISCODE'])
    df_feature = df_ID_SES_feature.loc[df_order.index]
    #GET THE INDEX LIST TO PRINT INTO FILE
    index_feature = df_feature.index

    FEATURE_array = df_feature[f'{feature}'].values
    
    FEATURE_list = [  #WE DO THIS BECAUSE THERE ARE 'NUM' STRING VALUES IN SOME FEATURES
    float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value
    for value in FEATURE_array
    ]
    
    if feature == 'ABETA': #IN ABETA THERE ARE '<NUM' VALUES
        FEATURE_list = [
        np.nan if isinstance(value, str) and value.startswith('>') and value[1:].isdigit() else value
        for value in FEATURE_list
        ]    
            
    print('_______________________________________________________________________________________')
    print(f' IN PLOT_LATENT_SPACE_{feature}: ¿Any NaN value in {feature}_list?: {np.isnan(FEATURE_list).any()}')
    
    z_list = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z_list]
    # Crear una lista de tuplas (z, feature) para mantener el orden
    paired_data = list(zip(z_list, FEATURE_list))

    # Filtrar las tuplas donde el segundo elemento no sea NaN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    z_clean, FEATURE_clean = zip(*filtered_data)
    print(f' IN PLOT_LATENT_SPACE_{feature}: ¿Any NaN value in {feature}_clean?: {np.isnan(FEATURE_clean).any()}')
    
    FEATURE_list_nonzero = np.where(np.array(FEATURE_clean) == 0, 1e-10, FEATURE_clean)

    
    #FOR DEBUGGING PURPOSES. SAVE INTO FILE PATH_ID AND ADNIMERGE_ID
    debugging = config['experiment']['debugging']
    
    if debugging == True:
        #TO SAVE THE LATENT SPACE OF EACH PATIENT OF TEST
        with open(os.path.join(results_folder, 'latent_space.txt'), 'w') as file:
            for item in z_list:
                file.write(f'{item}\n')
        #TO SAVE THE ID SES LIST ORDER OF THE LATENT SPACE
        with open(os.path.join(results_folder, 'latent_space_ID_SES_order.txt'), 'w') as file:
            for item in id_ses_list:
                file.write(f'{item}\n')
        
        #TO SAVE BOTH PATH OF ID AND feature INDEX AND SEE IF THEY MATCH
        path_test = extract_id_ses_from_path(path_test)
        test_path_ID_SES = merge_lists(path_test, id_ses_list)
        test_ADNIMERGE_path_ID_SES = merge_lists(test_path_ID_SES, index_feature)
        columns = ["test_path_ADNI_BIDS", "test_ADNIBIDS_ID_SES", "test_ADNIMERGE_ID_SES"]
        with open(os.path.join(results_folder, 'paths_ID_SES.txt'), 'w') as file:
            file.write('\t'.join(columns) + '\n')
            for item in test_ADNIMERGE_path_ID_SES:
                file.write(f'{item}\n')
        

    for latx in latents_plot:
        for laty in latents_plot:
            if latx != laty:
                dim_x = [item [latx] for item in z_clean] #GET DIM X OF LATENT SPACE
                dim_y = [item [laty] for item in z_clean] #GET DIM Y OF LATENT SPACE

                
                plt.figure(figsize = (8, 6))
                if feature == 'Ventricles':
                    plt.scatter(dim_x, dim_y, c = np.log(FEATURE_list_nonzero), cmap='viridis', edgecolors='k')
                else:
                    plt.scatter(dim_x, dim_y, c = FEATURE_list_nonzero, cmap='viridis', edgecolors='k')

                plt.xlabel(f'Latent dim {latx}')
                plt.ylabel(f'Latent dim {laty}')
                plt.title (f'Latent dim {latx} vs {laty}')

                plt.colorbar(label= f'{feature}')
                plt.savefig(f'{sub_folder}/scatter_plot_dim{latent_dim}_lat{latx}_lat{laty}.png', format='png', dpi=300, bbox_inches='tight')
                plt.close()
    return None




def global_glm_model(z_list, id_ses_list, df_ADNIMERGE, path_test, config, feature, results_folder):
    """
    This function fits a global Generalized Linear Model (GLM) to predict a given feature using latent variables 
    from `z_list`. It processes the data, fits the GLM, and saves the results, including the model summary.

    Steps:
    1. **Prepare data**:
       - Merges `df_ADNIMERGE` with `id_ses_list` to get a dataframe containing the feature to predict.
       - Sorts the dataset to match the order of `id_ses_list`.
    2. **Clean feature data**:
       - Processes the feature values, converting strings to numbers or NaN as necessary.
       - Special handling for features like 'ABETA' to replace values like '>' with NaN.
    3. **Filter out NaN values**: Filters out data points where the feature value is NaN.
    4. **Fit GLM model**:
       - Adds a constant to the latent variable data and fits a global GLM using the feature as the dependent variable.
    5. **Save results**:
       - Saves the GLM model summary to a text file.
       
    The function outputs the GLM model summary and stores it in a text file.
    """
    
    global_glm_folder = os.path.join(results_folder, 'global_glm_folder')
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    if not os.path.exists(global_glm_folder):
        os.makedirs(global_glm_folder)
    
    debugging = config['experiment']['debugging']
    
    latent_dim = config['model']['latent']
    
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, feature)
    df_ID_SES_feature = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_feature = df_ID_SES_feature.set_index(['PTID', 'VISCODE'])
    df_feature = df_ID_SES_feature.loc[df_order.index]
    #GET THE INDEX LIST TO PRINT INTO FILE
    index_feature = df_feature.index

    FEATURE_array = df_feature[f'{feature}'].values
    
    FEATURE_list = [ #CHANGE FORMAT FROM STRING TO FLOAT AND REMOVE '<NUM' AND '>NUM'
    np.nan if isinstance(value, str) and (value.startswith('<') or value.startswith('>')) and value[1:].isdigit() else float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value
    for value in FEATURE_array
    ]
    
    
    # if feature == 'ABETA': #IN ABETA THERE ARE '<NUM' VALUES
    #     FEATURE_list = [
    #     np.nan if isinstance(value, str) and value.startswith('>') and value[1:].isdigit() else value
    #     for value in FEATURE_list
    #     ]    
            
    print(f' IN GLOBAL_GLM_{feature}: ¿Any NaN value in {feature}_list?: {np.isnan(FEATURE_list).any()}')
    
    
    z = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z_list]
    
    # CREATE (Z, FEATURE) TUPLE TO KEEP ORDER
    paired_data = list(zip(z, FEATURE_list))

    # FILTER TUPLES WHERE SECOND ELEMENT IS NON NAN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    
    if debugging == True:
        with open(os.path.join(global_glm_folder, f'z_{feature}_list_filtered.txt'), 'w') as file:
            for item in filtered_data:
                file.write(f'{item}\n')
                
    z_clean, FEATURE_clean = zip(*filtered_data)
    FEATURE_list_nonzero = np.where(np.array(FEATURE_clean) == 0, 1e-10, FEATURE_clean)
    
    X = sm.add_constant(z_clean)
    
    glm_model = sm.GLM(FEATURE_list_nonzero, X, family = sm.families.Gaussian()).fit()
    
    with open('glm_model_summary.txt', 'w') as f:
        f.write(glm_model.summary().as_text())
    print('GLM summary saved in glm_model_summary.txt')
    
    
    
    
    
    
    
def glm_model(z_list, id_ses_list, df_ADNIMERGE, path_test, config, feature, results_folder):
    """
    This function fits a Generalized Linear Model (GLM) to predict a given feature using latent variables from `z_list`. 
    It processes the data, fits the GLM, and saves the results, including model summaries and plots.

    Steps:
    1. **Prepare data**:
       - Merge `df_ADNIMERGE` with `id_ses_list` to get a dataframe with IDs, visit codes, and the feature to predict.
       - Sort the data to match the order of the subject dataset (`id_ses_list`).
    2. **Clean data**:
       - Process the feature values to handle non-numeric strings or special cases like '>' (e.g., replace with NaN or valid numbers).
    3. **Filter out NaN values**: Remove any data points where the feature is NaN.
    4. **Fit GLM model**:
       - For each latent dimension, fit a GLM with the feature as the dependent variable and the latent variable as the independent variable.
       - Store the model summary and create a plot of the latent variable vs. the feature.
    5. **Save results**:
       - Save the GLM model summaries to a text file.
       - Save the plots showing the relationship between each latent dimension and the feature.
       
    The function outputs GLM model summaries and plots showing the relationship between each latent variable and the target feature.
    """
    
    single_glm_folder = os.path.join(results_folder, 'single_glm_folder')
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    if not os.path.exists(single_glm_folder):
        os.makedirs(single_glm_folder)
    
    debugging = config['experiment']['debugging']
    
    latent_dim = config['model']['latent']
    
    #MERGE ADNIMERGE DATASET WITH TEST_ID_SES AND GET DF (ID, SES, feature)
    df_ID_SES_feature = merge_id_ses_to_ADNIMERGE(id_ses_list, df_ADNIMERGE, feature)
    
    #SORT ADNIMERGE INTO SAME ORDER AS MY SUBJECT DATASET
    df_order = pd.DataFrame(id_ses_list, columns = ['PTID', 'VISCODE'])
    df_order.set_index(['PTID', 'VISCODE'], inplace=True)
    
    df_ID_SES_feature = df_ID_SES_feature.set_index(['PTID', 'VISCODE'])
    df_feature = df_ID_SES_feature.loc[df_order.index]
    #GET THE INDEX LIST TO PRINT INTO FILE
    index_feature = df_feature.index

    FEATURE_array = df_feature[f'{feature}'].values
    
    FEATURE_list = [ #CHANGE FORMAT FROM STRING TO FLOAT AND REMOVE '<NUM' AND '>NUM'
    np.nan if isinstance(value, str) and (value.startswith('<') or value.startswith('>')) and value[1:].isdigit() else float(value) if isinstance(value, str) and value.replace('.', '', 1).isdigit() else value
    for value in FEATURE_array
    ]
    
    
    # if feature == 'ABETA': #IN ABETA THERE ARE '<NUM' VALUES
    #     FEATURE_list = [
    #     np.nan if isinstance(value, str) and value.startswith('>') and value[1:].isdigit() else value
    #     for value in FEATURE_list
    #     ]        
    
    print(f' IN GLM_model_{feature}: ¿Any NaN value in {feature}_list?: {np.isnan(FEATURE_list).any()}')
    
    
    z = [tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in z_list]
    
    # CREATE (Z, FEATURE) TUPLE TO KEEP ORDER
    paired_data = list(zip(z, FEATURE_list))

    # FILTER TUPLES WHERE SECOND ELEMENT IS NON NAN
    filtered_data = [(x, y) for x, y in paired_data if not np.isnan(y)]
    
    if debugging == True:
        with open(os.path.join(single_glm_folder, f'z_{feature}_list_filtered.txt'), 'w') as file:
            for item in filtered_data:
                file.write(f'{item}\n')
                
    z_clean, FEATURE_clean = zip(*filtered_data)
    FEATURE_list_nonzero = np.where(np.array(FEATURE_clean) == 0, 1e-10, FEATURE_clean)
    
    model_summaries = []
    
    fig, axes = plt.subplots(latent_dim, 1, figsize=(8, 15))
    
    print(f"Shape of z_clean: {np.shape(z_clean)}")
    print(z_clean)

    
    for i in range(latent_dim):
        
        X = sm.add_constant(z_clean[:,i])
        glm_model = sm.GLM(FEATURE_list_nonzero, X, family = sm.families.Gaussian()).fit()
        
        model_summaries.append(glm_model.summary())
        
        predictions = glm_model.predict(X)
        axes[i].scatter(z_clean[:,i], FEATURE_clean, label = f'Latent {i}', color = 'b', alpha = 0.5)
        
        axes[i].plot(z_clean[:,i], predictions, color = 'r', label = 'GLM regression', linewidth = 2)
        
        axes[i].set_title(f'Relation between latent {i} and {feature}')
        axes[i].set_xlabel(f'latent {i}')
        axes[i].set_ylabel(f'{feature}')
        axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(f"GLM_latent_{i}_vs_{feature}.png")
        plt.close()
    
    
    
        
        
    with open('glm_model_summaries.txt', 'w') as f:
        for i, summary in enumerate(model_summaries):
            f.write(f"GLM summary for latent {i}:\n")
            f.write(summary.as_text())
            f.write("\n\n")
    