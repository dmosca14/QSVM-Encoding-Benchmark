import numpy as np
from sklearn.model_selection import train_test_split

def importa_dataset(dataset_scelto, numero_campioni, frazione_dati_per_train, frazione_dati_per_val, frazione_dati_per_test):

    # IMPORTAZIONE DEI DATI.

    # "np.asarray" permette di ricevere in ingresso anche un tipo di dataset "fetch", cioè un foglio tipo Excel, anziché un "load", cioè un array puro. 
    # "np.asarray" converte poi il dataset in un array. Per fare questo, bisogna anche aggiungere a y_grezzo "astype(int)", per evitare che gli
    # attributi siano stringhe testuali e non 0 ed 1, come vogliamo che siano.

    X_grezzo = np.asarray(dataset_scelto.data)
    y_grezzo = np.asarray(dataset_scelto.target).astype(int) 
    
    numero_campioni_totale = X_grezzo.shape[0] # Numero di righe del dataset, quindi numero di campioni del dataset.

    # RIDUZIONE DEL DATASET TOTALE.

    # Vogliamo estrarre un sottoinsieme di campioni, chiamato X_ridotto, che contiene un numero di campioni pari a n_campioni, dal dataset totale.
    # Aggiungiamo anche un check che controlli che il numero di campioni da estrarre non sia inferiore al numero di campioni totale.
    
    if numero_campioni >= numero_campioni_totale:
        raise ValueError(f"Il numero di campioni richiesto, {numero_campioni}, è maggiore o uguale al totale del numero di campioni nel dataset, {numero_campioni_totale}.")
    
    print("Il numero di campioni da estrarre desiderato è valido.")
    
    X_grezzo_ridotto, _, y_grezzo_ridotto, _ = train_test_split(
        X_grezzo, y_grezzo, 
        train_size = numero_campioni, 
        stratify = y_grezzo, 
    )

    # CREAZIONE DEL TRAINING SET, DEL VALIDATION SET E DEL TEST SET.

    # Vogliamo dividere X_ridotto in tre sottoinsiemi: quello di training (train set), quello di validation (val set) e quello di test (set test), sulla base
    # dei rapporti frazione_dati_per_train, frazione_dati_per_val e frazione_dati_per_test. Per prima cosa, controlliamo che questi rapporti siano validi, 
    # cioè sommmino ad 1.
    
    somma_frazioni = frazione_dati_per_train + frazione_dati_per_val + frazione_dati_per_test
    
    if not np.isclose(somma_frazioni, 1.0):
        raise ValueError(f"La somma delle frazioni è diversa da 1" )
    
    print("I rapporti scelti sono validi.")

   # Per dividere X_ridotto in tre parti, utilizziamo la funzione train_test_split, che però divide un dataset in sole due parti; 
   # dovremo quindi ripetere il processo per due volte, una per separare X_ridotto in X_train e X_val_e_test (quello che rimane da 
   # X_ridotto avendo tolto X_train) e una per separare X_val_e_test in X_val da X_test. 
    
    X_train_importato, X_val_e_test_grezzo, y_train_importato, y_val_e_test_grezzo = train_test_split(
        X_grezzo_ridotto, y_grezzo_ridotto, 
        train_size = frazione_dati_per_train, 
        stratify = y_grezzo_ridotto, 
    )
    
    X_val_importato, X_test_importato, y_val_importato, y_test_importato = train_test_split(
        X_val_e_test_grezzo, y_val_e_test_grezzo, 
        test_size = frazione_dati_per_val/(frazione_dati_per_val + frazione_dati_per_test), 
        stratify = y_val_e_test_grezzo, 
    )

    # ESPORTAZIONE DEI DATI.

    # I vari dataset elaborati, in questo codice, verranno sempre importati ed esportati secondo la seguente formula: una tupla che contiene le X e le y
    # per ogni tipo di dataset (train, val, set), e una tupla finale, chiamata "set_aggettivo", che contiene le tre tuple precedenti. In questo modo
    # si mantiene l'accesso ad ogni tipo di dato durante tutto il codice, e si ha un sistema coerente. 
    
    train_set_importato = (X_train_importato, y_train_importato)
    val_set_importato =(X_val_importato, y_val_importato)
    test_set_importato = (X_test_importato, y_test_importato)

    set_importato = (train_set_importato, val_set_importato, test_set_importato)
    
    return set_importato