import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def prepara_dataset(set_importato, numero_features):

    # IMPORTAZIONE DEI DATI.

    train_set_importato, val_set_importato, test_set_importato = set_importato

    X_train_importato, y_train_importato = train_set_importato
    X_val_importato, y_val_importato = val_set_importato
    X_test_importato, y_test_importato = test_set_importato
    
    numero_features_originali_dataset = X_train_importato.shape[1]

    # Se abbiamo richiesto un numero di features superiore a quello del dataset, il codice dovrà dare errore.
    
    if numero_features > numero_features_originali_dataset:

        raise ValueError(f"Il numero di features richiesto, {numero_features}, è maggiore della quantità totale di features presenti nel dataset, {numero_features_originali_dataset}.")

    if numero_features == numero_features_originali_dataset:
        print(f"Il numero di features richiesto, {numero_features}, è uguale alla quantità totale di features presenti nel dataset.")

    else:
        print(f"Il numero di features richiesto, {numero_features}, è minore della quantità totale di features presenti nel dataset, {numero_features_originali_dataset}.")

    # APPLICAZIONE DELLA PCA.

    # Se abbiamo richiesto un numero di features inferiore o uguale a quello del dataset, il codice dovrà applicare la PCA, dicendoci anche quanta 
    # informazione è stata compressa. Prima di applicare la PCA, bisogna riscalare i dati in modo da fargli avere media nulla e varianza unitaria. 
    # Per fare questo, utilizziamo StandardScaler. La media e la varianza verranno calcolate utilizzando soltanto il train set (fit), per evitare il 
    # data leakage, ma trasformeremo tutti i tre set (transform). Quindi, al train set si applica fit_transform, che fa entrambe le cose, mentre 
    # agli altri set solamente transform.

    scaler = StandardScaler() 
    X_train_scalato = scaler.fit_transform(X_train_importato) 
    X_val_scalato = scaler.transform(X_val_importato) 
    X_test_scalato = scaler.transform(X_test_importato)
            
    pca = PCA(n_components = numero_features)
    X_train_preparato = pca.fit_transform(X_train_scalato)
    X_val_preparato = pca.transform(X_val_scalato)
    X_test_preparato = pca.transform(X_test_scalato)
            
    varianza_conservata = np.sum(pca.explained_variance_ratio_) # informazione conservata dopo la PCA
    print(f"PCA eseguita con successo. Informazione conservata: {varianza_conservata * 100:.2f}%")

    # ESPORTAZIONE DEI DATI
            
    train_set_preparato = (X_train_preparato, y_train_importato)
    val_set_preparato = (X_val_preparato, y_val_importato)
    test_set_preparato = (X_test_preparato, y_test_importato)

    set_preparato = (train_set_preparato, val_set_preparato, test_set_preparato)

    return set_preparato