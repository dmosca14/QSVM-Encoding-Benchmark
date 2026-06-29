from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pennylane as qml
import time

nome_encoding = "IQP Encoding"

def funzione_adattamento_dati(set_preparato):

    # IMPORTAZIONE DEI DATI.

    train_set_preparato, val_set_preparato, test_set_preparato = set_preparato

    X_train_preparato, y_train_preparato = train_set_preparato
    X_val_preparato, y_val_preparato = val_set_preparato
    X_test_preparato, y_test_preparato = test_set_preparato

    # ADATTAMENTO DEL DATASET ALL'ENCODING.

    # Per adattare il dataset preparato con PCA all'IQP encoding, abbiamo bisogno di rendere la codifica biunivoca: dobbiamo mappare i valori delle 
    # features dei campioni fra -1 e 1, usando MinMaxScaler. Non facciamo [0, pi] perché nell'IQP encoding le fasi vengono poi moltiplicate, 
    # e restringere a [-1, 1] ci assicura che anche il prodotto delle fasi rimarrà fra [-1, 1], evitando che le fasi comincino a girare
    # di tanti periodi, perdendo la distanza fra i punti. Nell'angle encoding questo non è necessario perché le fasi non vengono mai moltiplicate, 
    # quindi possiamo restringerci senza problemi a [0, pi] che è l'intervallo massimo possibile, aumentando l'espressività. Calcoliamo i parametri 
    # con cui riscalare (fit) solo sul train set per evitare il data leakage, limitandoci a trasformare (transform) gli altri due set. Tuttavia, dato 
    # che i campioni di validation e test potrebbero avere valori originali più estremi rispetto al train e finire "fuori scala", applichiamo 
    # un clipping con np.clip.

    riscalatore = MinMaxScaler(feature_range = (-1, 1))
    
    X_train_scalato = riscalatore.fit_transform(X_train_preparato)
    X_val_scalato = riscalatore.transform(X_val_preparato)
    X_test_scalato = riscalatore.transform(X_test_preparato)
    
    X_val_scalato_e_clippato = np.clip(X_val_scalato, -1, 1) 
    X_test_scalato_e_clippato = np.clip(X_test_scalato, -1, 1)

    # Controlliamo quanti campioni sono stati effettivamente clippati, per valutare quanti outlier anomali erano presenti. Se il numero di campioni 
    # clippati supera una certa soglia, c'è un allarme. In quel caso, basta eseguire tutto da capo, in quanto verranno casualmente estratti dei nuovi
    # train set, validation set e test set, che nella maggior parte dei casi ridistribuiranno i dati in maniera migliore e non si avrà questo problema.

    numero_campioni_clippati_val = np.sum(X_val_scalato != X_val_scalato_e_clippato) 
    numero_campioni_clippati_test = np.sum(X_test_scalato != X_test_scalato_e_clippato)
    
    totale_campioni_val = X_val_scalato.size 
    totale_campioni_test = X_test_scalato.size
    
    frazione_clippati_val = (numero_campioni_clippati_val / totale_campioni_val)
    frazione_clippati_test = (numero_campioni_clippati_test / totale_campioni_test)

    if frazione_clippati_val > 0.05:

        raise ValueError(f"Anomalie nel validation set, il {frazione_clippati_val*100:.2f}% dei dati è stato clippato (fuori scala rispetto al train set).")
        
    if frazione_clippati_test > 0.05:
        
        raise ValueError(f"Anomalie nel test set, il {frazione_clippati_test*100:.2f}% dei dati è stato clippato (fuori scala rispetto al train set).")
    
    # ESPORTAZIONE DEI DATI.

    train_set_adattato = (X_train_scalato, y_train_preparato)
    val_set_adattato = (X_val_scalato_e_clippato, y_val_preparato)
    test_set_adattato = (X_test_scalato_e_clippato, y_test_preparato)

    set_adattato = (train_set_adattato, val_set_adattato, test_set_adattato)

    return set_adattato


def kernel(set_preparato, numero_features, numero_ripetizioni, pattern):

    # IMPORTAZIONE DEI DATI.

    set_adattato = funzione_adattamento_dati(set_preparato)

    train_set_adattato, val_set_adattato, test_set_adattato = set_adattato

    X_train_adattato, _ = train_set_adattato
    X_val_adattato, _ = val_set_adattato
    X_test_adattato, _ = test_set_adattato

    # DEFINIZIONE DELLA FUNZIONE DI KERNEL E CALCOLO DELLE MATRICI DI GRAM.

    numero_qubits_IQP = numero_features
    dev_IQP = qml.device("lightning.qubit", wires = numero_qubits_IQP)

    # Definizione della funzione che fa l'encoding e calcola il kernel, con la tecnica compute - uncompute e proiettando sullo stato <00,...,0|.
    
    @qml.qnode(dev_IQP)
    def circuito_quantistico(campione_1, campione_2): 

        # Qui compare l'iperparametro quantistico di questo encoding n_repeats, che controlla l'espressività del circuito.

        qml.IQPEmbedding(features = campione_1, wires = range(numero_qubits_IQP), n_repeats = numero_ripetizioni, pattern = pattern)
        qml.adjoint(qml.IQPEmbedding)(features = campione_2, wires = range(numero_qubits_IQP), n_repeats = numero_ripetizioni, pattern = pattern) 
        stato_zero = np.zeros(numero_qubits_IQP) 
        return qml.expval(qml.Projector(stato_zero, wires = range(numero_qubits_IQP)))
    
    # Per la matrice di Gram calcolata sul train set, utilizziamo square_kernel_matrix in modo da snellire il calcolo; questa, infatti, 
    # pone già ad 1 i termini sulla diagonale, e calcola solo uno dei due triangoli della matrice, duplicando poi sull'altro. Per le altre due matrici
    # di Gram, invece, non si può fare, dato che non stiamo calcolando un set contro sé stesso, ma due set diversi. 

    matrice_gram_train = qml.kernels.square_kernel_matrix(X_train_adattato, circuito_quantistico, assume_normalized_kernel = True)
    matrice_gram_val = qml.kernels.kernel_matrix(X_val_adattato, X_train_adattato, circuito_quantistico)
    matrice_gram_test = qml.kernels.kernel_matrix(X_test_adattato, X_train_adattato, circuito_quantistico)

    # ESPORTAZIONE DELLE MATRICI DI GRAM E DEI DATI. 

    matrici_gram = matrice_gram_train, matrice_gram_val, matrice_gram_test

    return set_adattato, matrici_gram