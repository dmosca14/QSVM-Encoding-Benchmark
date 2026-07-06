import sys
import os
from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import time
os.environ["OMP_NUM_THREADS"] = "4"

qml_code_dir = Path.cwd().parent
sys.path.append(str(qml_code_dir))

from sklearn.datasets import make_circles, make_moons, make_classification, load_breast_cancer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from moduli.preprocessing_dataset.preprocessing_dataset import preprocessa_dataset
from moduli.preprocessing_dataset.preparazione_dataset import prepara_dataset
from moduli.qsvm.qsvm import costruisci_qsvm
from moduli.qsvm.configurazione_cartella_run import cartella_run_attuale
from moduli.kernel import kernel_amplitude, kernel_angle, kernel_basis, kernel_IQP, kernel_rbf, kernel_projected
from moduli.metriche import kernel_target_alignment, coefficiente_geometrico

# Configurazione generale

numero_campioni = 200
numero_features = 4

frazione_dati_per_train = 0.80  
frazione_dati_per_test = 0.20
griglia_C = [1.0, 10.0, 100.0, 1000.0]


def genera_dataset_ad_hoc(n_samples=200):
    np.random.seed(42)
    X = np.random.uniform(-np.pi, np.pi, (n_samples, 2))
    y = np.array([1 if np.sin(x[0]) * np.cos(x[1]) > 0 else 0 for x in X])
    return X, y

def genera_dataset_xor_alta_dim(n_samples=200, n_features=numero_features, random_state=1):
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1, 1, (n_samples, n_features))
    segni = np.sign(X)
    y = (np.prod(segni, axis=1) > 0).astype(int)
    return X, y


dati_breast_cancer = load_breast_cancer()
def sottocampiona_breast_cancer(n_samples=numero_campioni, random_state=1):
    X, y = dati_breast_cancer.data, dati_breast_cancer.target
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n_samples, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


datasets = {
    "Linear": make_classification(n_samples=numero_campioni, n_features=numero_features, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1),
    "Circles": make_circles(n_samples=numero_campioni, noise=0.08, factor=0.3, random_state=1),
    "Moons": make_moons(n_samples=numero_campioni, noise=0.1, random_state=1),
    "Quantum_Ad_Hoc": genera_dataset_ad_hoc(n_samples=numero_campioni),
    "XOR_Alta_Dim": genera_dataset_xor_alta_dim(n_samples=numero_campioni, n_features=numero_features),
    "Breast_Cancer": sottocampiona_breast_cancer(n_samples=numero_campioni)
}


def costruisci_configurazioni_encoding(pattern_catena, pattern_tutti):
    return [
        {"nome": "Amplitude", "modulo": kernel_amplitude, "ip_quantistici": {}},
        {"nome": "Angle", "modulo": kernel_angle, "ip_quantistici": {}},
        {"nome": "Basis", "modulo": kernel_basis, "ip_quantistici": {"tau_bit_per_feature": [1, 2]}},
        {"nome": "IQP", "modulo": kernel_IQP, "ip_quantistici": {"numero_ripetizioni": [1, 2], "pattern": [pattern_catena, pattern_tutti]}},
        {"nome": "Projected", "modulo": kernel_projected, "ip_quantistici": {"gamma": [0.1, 1.0, 5.0], "numero_layer": [1, 2]}}
    ]


risultati_globali = []

# Loop di esecuzione

for nome_ds, data in datasets.items():
    print(f"\n--- Elaborazione: {nome_ds} ---")

    # Estrazione dataset
    
    train_set_raw, test_set_raw = preprocessa_dataset(data, frazione_dati_per_train)
    y_train, y_test = train_set_raw[1], test_set_raw[1]

    # Standardizzazione e PCA per RBF 
    
    train_set_pca, test_set_pca, num_features = prepara_dataset(train_set_raw, test_set_raw, numero_features)

    # Pattern per IQP
    
    pattern_catena = [[i, i + 1] for i in range(num_features - 1)]
    pattern_tutti = [list(coppia) for coppia in itertools.combinations(range(num_features), 2)]
    configurazioni_encoding = costruisci_configurazioni_encoding(pattern_catena, pattern_tutti)

    # Kernel RBF 
    
    _, matrici_gram_rbf = kernel_rbf.kernel(train_set_pca, test_set_pca)
    K_classico_train = matrici_gram_rbf[0]
    kta_rbf = kernel_target_alignment(K_classico_train, y_train)

    for config in configurazioni_encoding:
        # UNA SOLA chiamata per encoding: passiamo l'intera griglia di iperparametri
        # quantistici, ed è addestramento() a fare la grid-search completa
        # (quantistici x classici) e a scegliere/restituire solo il modello migliore.
        modello, set_adattato, matrici_gram, storico, ip_q_migliori, ip_c_migliori = costruisci_qsvm(
            train_set_raw, test_set_raw, num_features, config["modulo"],
            config["ip_quantistici"], {"C": griglia_C}, nome_ds
        )

        K_train_q, K_test_q = matrici_gram

        risultati_globali.append({
            "Dataset": nome_ds,
            "Encoding": config["nome"],
            "Parametri Q": str(ip_q_migliori),
            "C": ip_c_migliori.get("C", modello.C),
            "RBF KTA": f"{kta_rbf:.4f}",
            "Quantum KTA": f"{kernel_target_alignment(K_train_q, y_train):.4f}",
            "Coeff. Geom": f"{coefficiente_geometrico(K_classico_train, K_train_q):.4f}",
            "Acc": f"{accuracy_score(y_test, modello.predict(K_test_q)):.4f}",
            "F1": f"{f1_score(y_test, modello.predict(K_test_q), average='weighted'):.4f}"
        })

    pd.DataFrame(risultati_globali).to_csv(os.path.join(cartella_run_attuale, "risultati_parziali.csv"), index=False)

# Output

df = pd.DataFrame(risultati_globali)
print(df.to_string(index=False))

percorso_csv = os.path.join(cartella_run_attuale, "risultati_globali.csv")
df.to_csv(percorso_csv, index=False)
