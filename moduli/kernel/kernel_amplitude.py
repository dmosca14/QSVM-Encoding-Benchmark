from sklearn.preprocessing import normalize
import numpy as np
import pennylane as qml
nome_encoding = "Amplitude Encoding"

def funzione_adattamento_dati(set_preparato):

    # IMPORTAZIONE DEI DATI

    train_set_preparato, val_set_preparato, test_set_preparato = set_preparato

    X_train_preparato, y_train_preparato = train_set_preparato
    X_val_preparato, y_val_preparato = val_set_preparato
    X_test_preparato, y_test_preparato = test_set_preparato

    # ADATTAMENTO DEL DATASET ALL'ENCODING.

    # Per adattare il dataset preparato con PCA all'amplitude encoding, dobbiamo assicurarci che i vettori contenenti le
    # features dei campioni che passiamo alla funzione di encoding abbiano norma unitaria.

    X_train_scalato = normalize(X_train_preparato, norm = "l2")
    X_val_scalato = normalize(X_val_preparato, norm = "l2")
    X_test_scalato = normalize(X_test_preparato, norm = "l2")

    # ESPORTAZIONE DEI DATI

    train_set_adattato = (X_train_scalato, y_train_preparato)
    val_set_adattato = (X_val_scalato, y_val_preparato)
    test_set_adattato = (X_test_scalato, y_test_preparato)

    set_adattato = (train_set_adattato, val_set_adattato, test_set_adattato)

    return set_adattato


def kernel(set_preparato, numero_features):

    # IMPORTAZIONE DEI DATI

    set_adattato = funzione_adattamento_dati(set_preparato)

    train_set_adattato, val_set_adattato, test_set_adattato = set_adattato

    X_train_adattato, _ = train_set_adattato
    X_val_adattato, _ = val_set_adattato
    X_test_adattato, _ = test_set_adattato

    # DEFINIZIONE DELLA FUNZIONE DI KERNEL E CALCOLO DELLE MATRICI DI GRAM.

    numero_qubits_amplitude = int(np.ceil(np.log2(numero_features)))
    dev_amplitude = qml.device("lightning.qubit", wires = numero_qubits_amplitude)

    # Definizione della funzione che fa l'encoding e calcola il kernel, con la tecnica compute - uncompute e proiettando sullo stato <00,...,0|.

    @qml.qnode(dev_amplitude)
    def circuito_quantistico(campione_1, campione_2): 

        # Dobbiamo inserire normalize = True in modo tale da essere sicuri che i calcoli numerici non abbiano modificato la norma unitaria dei vettori 
        # che rappresentano i campioni, altrimenti non sarebbero più vettori validi per la meccanica quantistica.

        qml.AmplitudeEmbedding(features = campione_1, wires = range(numero_qubits_amplitude), normalize = True, pad_with = 0.)
        qml.adjoint(qml.AmplitudeEmbedding)(features = campione_2, wires = range(numero_qubits_amplitude), normalize = True, pad_with = 0.) 
        stato_zero = np.zeros(numero_qubits_amplitude) 
        return qml.expval(qml.Projector(stato_zero, wires = range(numero_qubits_amplitude)))

    # Per la matrice di Gram calcolata sul train set, utilizziamo square_kernel_matrix in modo da snellire il calcolo; questa, infatti, 
    # pone già ad 1 i termini sulla diagonale, e calcola solo uno dei due triangoli della matrice, duplicando poi i risultatisull'altro. 
    # Per le altre due matrici di Gram, invece, non si può fare, dato che non stiamo calcolando un set contro sé stesso, ma due set diversi. 

    matrice_gram_train = qml.kernels.square_kernel_matrix(X_train_adattato, circuito_quantistico, assume_normalized_kernel = True)
    matrice_gram_val = qml.kernels.kernel_matrix(X_val_adattato, X_train_adattato, circuito_quantistico)
    matrice_gram_test = qml.kernels.kernel_matrix(X_test_adattato, X_train_adattato, circuito_quantistico)

    # ESPORTAZIONE DELLE MATRICI DI GRAM E DEI DATI. 

    matrici_gram = matrice_gram_train, matrice_gram_val, matrice_gram_test

    return set_adattato, matrici_gram