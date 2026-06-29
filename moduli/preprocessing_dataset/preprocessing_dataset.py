from . import importazione_dataset
from . import preparazione_dataset

def preprocessa_dataset(dataset_scelto, numero_campioni, frazione_dati_per_train, frazione_dati_per_val, frazione_dati_per_test, numero_features):

    print("\nPREPROCESSING DATASET\n")

    # IMPORTAZIONE DEL DATASET E CREAZIONE DEL TRAINING SET, DEL VALIDATION SET E DEL TEST SET.

    set_importato = importazione_dataset.importa_dataset(
        dataset_scelto = dataset_scelto,
        numero_campioni = numero_campioni,
        frazione_dati_per_train = frazione_dati_per_train,
        frazione_dati_per_val = frazione_dati_per_val,
        frazione_dati_per_test = frazione_dati_per_test
    )
     
    # PREPARAZIONE DEL DATASET TRAMITE PCA. 
    
    set_preparato = preparazione_dataset.prepara_dataset(
        set_importato = set_importato,
        numero_features = numero_features
    )

    print("Preprocessing del dataset completato.\n")

    return set_preparato