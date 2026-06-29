import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

def test_e_visualizzazione(nome_encoding, set_adattato, matrici_gram, modello, nome_cartella):

    # ESTRAZIONE DEI DATI.

    train_set_adattato, _, test_set_adattato = set_adattato
    _, y_train_adattato = train_set_adattato
    _, y_test_adattato = test_set_adattato
    matrice_gram_train, _, matrice_gram_test = matrici_gram

   # TEST DEL MODELLO SUL TEST SET. 
    
    predizioni_test = modello.predict(matrice_gram_test)
    report_dict = classification_report(y_test_adattato, predizioni_test, output_dict = True, zero_division = 0)

    # Da qui in poi, essendo una parte puramente fatta di grafici, il codice è stato fatto interamente da Gemini.

    tabella = []
    for chiave, valori in report_dict.items():
        if chiave in ["accuracy", "macro avg", "weighted avg"]: 
            continue 
        tabella.append([
            f"Classe {chiave}", 
            f"{valori['precision'] * 100:.2f}%", 
            f"{valori['recall'] * 100:.2f}%", 
            f"{valori['f1-score'] * 100:.2f}%", 
            int(valori['support'])
        ])

    tabella.append([
        "Medie (Macro)", 
        f"{report_dict['macro avg']['precision'] * 100:.2f}%", 
        f"{report_dict['macro avg']['recall'] * 100:.2f}%", 
        f"{report_dict['macro avg']['f1-score'] * 100:.2f}%", 
        int(report_dict['macro avg']['support'])
    ])

    intestazioni = ["", "Precision", "Recall", "F1-Score", "Campioni realmente presenti"]
    tabella_formattata = tabulate(tabella, headers=intestazioni, tablefmt="fancy_grid", stralign="center", numalign="center")
    
    print(f"\nDIAGNOSI FINALE SUL TEST SET:\n")
    print(tabella_formattata)

    matrice_di_confusione = confusion_matrix(y_test_adattato, predizioni_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(matrice_di_confusione, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax1, annot_kws={"size": 14})
    ax1.set_title(f"Matrice di confusione\n{nome_encoding}", fontsize=14)
    ax1.set_xlabel("Predizione della QSVM", fontsize=12)
    ax1.set_ylabel("Classe vera", fontsize=12)

    indici_ordinati = np.argsort(y_train_adattato)
    matrice_gram_ordinata = matrice_gram_train[indici_ordinati, :][:, indici_ordinati]

    sns.heatmap(matrice_gram_ordinata, cmap="viridis", vmin=0, vmax=1, ax=ax2, cbar=True)
    ax2.set_title(f"Matrice di Gram (ordinata per classe)\n{nome_encoding}", fontsize=14)
    ax2.set_xlabel("Indici riordinati", fontsize=12)
    ax2.set_ylabel("Indici riordinati", fontsize=12)

    plt.tight_layout()
    percorso_grafico = os.path.join(nome_cartella, "grafici_modello_ottimo.svg")
    plt.savefig(percorso_grafico, dpi=300)
    plt.show()

    return tabella_formattata