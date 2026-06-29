import os
from datetime import datetime
import zoneinfo # Devo mettere questo import perché utilizzando un Container mi prende l'orario di Greenwich e non quello di Roma. 

# CREAZIONE DELLA CARTELLA DOVE SALVARE I RISULTATI DEI VARI RUN. 

fuso_orario_locale = zoneinfo.ZoneInfo("Europe/Rome")
timestamp_ora = datetime.now(fuso_orario_locale)

cartella_primaria = "risultati_run"
cartella_giorno = timestamp_ora.strftime(f"%Y%m%d")
cartella_giorno_e_orario = timestamp_ora.strftime(f"%Y%m%d_%H%M%S") 

cartella_run_attuale = os.path.join(cartella_primaria, cartella_giorno, cartella_giorno_e_orario)

os.makedirs(cartella_run_attuale, exist_ok = True) # exist_ok = True crea una cartella se non esiste e non fa nulla altrimenti.