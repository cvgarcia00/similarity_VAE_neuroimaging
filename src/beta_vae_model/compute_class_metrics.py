import pandas as pd
import glob
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, classification_report, f1_score, precision_score

# Lists for saving the metrics for each iteration
accuracy_list = []
sensitivity_list = []
specificity_list = []
balanced_accuracy_list = []

file_paths = glob.glob('y_true_vs_y_pred_ADvsCN*.csv')

for file_path in file_paths:
    print(f"Procesando archivo: {file_path}")

    df = pd.read_csv(file_path)
    y_true = df['y_true'].astype(int).tolist()
    y_pred = df['y_pred'].astype(int).tolist()

    # Metrics for this iteration
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)  # Sensitivity (recall de AD)
    spec = recall_score(y_true, y_pred, pos_label=0)  # Specificity (recall de CN)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # We save the metrics
    accuracy_list.append(acc)
    sensitivity_list.append(sens)
    specificity_list.append(spec)
    balanced_accuracy_list.append(bal_acc)

    # Individual report (optional)
    print(classification_report(df['y_true'], df['y_pred'], target_names=['CN', 'AD']))

# We transform the lists into numpy arrays to compute mean and std
accuracy_list = np.array(accuracy_list)
sensitivity_list = np.array(sensitivity_list)
specificity_list = np.array(specificity_list)
balanced_accuracy_list = np.array(balanced_accuracy_list)

# We print mean and std
print(f"\n=== Average results and std ===")
print(f"Accuracy: {accuracy_list.mean():.4f} ± {accuracy_list.std():.4f}")
print(f"Sensitivity: {sensitivity_list.mean():.4f} ± {sensitivity_list.std():.4f}")
print(f"Specificity: {specificity_list.mean():.4f} ± {specificity_list.std():.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_list.mean():.4f} ± {balanced_accuracy_list.std():.4f}")
