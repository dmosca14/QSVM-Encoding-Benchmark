import os
from .configurazione_cartella_run import cartella_run_attuale
from .addestramento import addestramento
from .visualizzazione import test_e_visualizzazione

def costruisci_qsvm(set_preparato, numero_features, modulo_encoding, iperparametri_quantistici, iperparametri_classici):

    # CREAZIONE DELLA CARTELLA RELATIVA ALLA RUN ATTUALE.

    nome_encoding = getattr(modulo_encoding, "nome_encoding")
    nome_pulito_encoding = nome_encoding.replace(" ", "_").lower() # Serve per trasformare, ad esempio, "Basis Encoding" in "basis_encoding".
   
    # Estraiamo l'orario dal nome della cartella della run attuale.

    timestamp_run = os.path.basename(cartella_run_attuale)

    # Con l'orario giusto, creiamo una cartella relativa ad ogni encoding. L'orario della run si riferisce all'avvio del codice globale,
    # non a quello di ogni encoding. 

    nome_cartella_encoding = f"{nome_pulito_encoding}_{timestamp_run}"
    
    # Assembliamo il percorso completo e creiamo l'ultima cartella, con tale percorso.

    percorso_finale = os.path.join(cartella_run_attuale, nome_cartella_encoding)
    os.makedirs(percorso_finale, exist_ok = True)

    # COSTRUZIONE DELLA QSVM.

    print(f"QUANTUM SUPPORT VECTOR MACHINE PER {nome_encoding.upper()}")
    print(f"\nDestinazione output: {percorso_finale}")

    # Addestriamo e validiamo la QSVM.

    set_adattato, matrici_gram, modello, storico = addestramento(
        nome_encoding, set_preparato, numero_features, modulo_encoding, 
        iperparametri_quantistici, iperparametri_classici, percorso_finale
    )

    # Testiamo la miglior QSVM e visualizziamone i risultati. 

    test_e_visualizzazione(nome_encoding, set_adattato, matrici_gram, modello, percorso_finale)

    return modello, set_adattato