import pandas as pd
import glob
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, classification_report, f1_score, precision_score

# Listas para almacenar las métricas de cada repetición
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

    # Métricas para esta repetición
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)  # Sensitivity (recall de AD)
    spec = recall_score(y_true, y_pred, pos_label=0)  # Specificity (recall de CN)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Guardamos las métricas en listas
    accuracy_list.append(acc)
    sensitivity_list.append(sens)
    specificity_list.append(spec)
    balanced_accuracy_list.append(bal_acc)

    # Reporte individual (opcional)
    print(classification_report(df['y_true'], df['y_pred'], target_names=['CN', 'AD']))

# Convertimos las listas a arrays de numpy para poder calcular media y std
accuracy_list = np.array(accuracy_list)
sensitivity_list = np.array(sensitivity_list)
specificity_list = np.array(specificity_list)
balanced_accuracy_list = np.array(balanced_accuracy_list)

# Imprimimos el promedio y la desviación estándar
print(f"\n=== Resultados promedio y desviación estándar ===")
print(f"Accuracy: {accuracy_list.mean():.4f} ± {accuracy_list.std():.4f}")
print(f"Sensitivity: {sensitivity_list.mean():.4f} ± {sensitivity_list.std():.4f}")
print(f"Specificity: {specificity_list.mean():.4f} ± {specificity_list.std():.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_list.mean():.4f} ± {balanced_accuracy_list.std():.4f}")