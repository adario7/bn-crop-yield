import pandas as pd

# Carica il file specificando il separatore corretto
df = pd.read_csv('data/crops.csv', sep=';', on_bad_lines='warn')


print(df.columns.to_list())  # Controlla ora le colonne corrette

# Trova l'indice della colonna 'OBS_STATUS'
col_index = df.columns.get_loc('OBS_STATUS')

# Seleziona tutte le colonne fino a 'OBS_STATUS' esclusa
df_filtered = df.iloc[:, :col_index]

# Salva il dataframe filtrato
df_filtered.to_csv('dataset_filtrato.csv', index=False, sep=';')
