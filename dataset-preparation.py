import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carica i dati
df_raccolto = pd.read_csv("build/dataset_filtrato.csv", delimiter=';')
df_clima = pd.read_csv("build/environment.csv", delimiter=';')

# Colonne da rimuovere da df_clima
col_da_rimuovere = [
    'Pioggia_10AMB003', 'Pioggia_10AMB008', 'Pioggia_10AMB011', 'Pioggia_10AMB012',
    'Pioggia_10AMB014', 'Pioggia_10AMB016', 'Pioggia_10AMB017', 'Pioggia_10AMB018P',
    'Pioggia_10AMB024P', 'Pioggia_10AMB025', 'Pioggia_10AMB026', 'Pioggia_10AMB027'
]

df_clima = df_clima.drop(columns=[c for c in col_da_rimuovere if c in df_clima.columns])

# Rimuovi colonne con >50% NaN e riempi i NaN rimanenti con forward fill
df_clima = df_clima.dropna(axis=1, thresh=len(df_clima)*0.5)
df_clima = df_clima.ffill()

# Discretizza temperatura e precipitazioni
df_clima['Temp_cat'] = pd.cut(df_clima['Temperature_Mean_C'], bins=3, labels=['bassa', 'media', 'alta'])
df_clima['Pioggia_cat'] = pd.cut(df_clima['Precipitation_Annual_mm'], bins=4, labels=['molto_bassa', 'bassa', 'media', 'alta'])

# Label Encoding per le colonne categoriche di df_clima
categorical_cols = ['Province', 'Temp_cat', 'Pioggia_cat']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clima[col] = le.fit_transform(df_clima[col].astype(str))
    label_encoders[col] = le

# Assicurati che 'Territorio' in df_raccolto sia stringa
df_raccolto['Territorio'] = df_raccolto['Territorio'].astype(str)

# Estendi l'encoder per Province per includere tutte le province di df_raccolto
le_province = label_encoders['Province']
missing_provinces = set(df_raccolto['Territorio'].unique()) - set(le_province.classes_)
if missing_provinces:
    le_province.classes_ = np.append(le_province.classes_, list(missing_provinces))

# Trasforma la colonna Territorio in codici numerici compatibili
df_raccolto['Territorio_encoded'] = le_province.transform(df_raccolto['Territorio'])

# Trova province mancanti in df_clima
province_clima_set = set(df_clima['Province'].unique())
province_raccolto_set = set(df_raccolto['Territorio_encoded'].unique())
province_mancanti = province_raccolto_set - province_clima_set

# Per ogni provincia mancante, crea righe con valori medi per ogni anno presente in df_clima
for provincia_cod in province_mancanti:
    for anno in df_clima['Year'].unique():
        df_media = df_clima[df_clima['Year'] == anno].select_dtypes(include=[np.number]).mean()
        nuova_riga = df_media.to_dict()
        nuova_riga['Province'] = provincia_cod
        nuova_riga['Year'] = anno
        # Per colonne categoriche, usa valore medio o -1 se non numeriche (già numeriche perché encoded)
        df_clima = pd.concat([df_clima, pd.DataFrame([nuova_riga])], ignore_index=True)

# Fai il merge usando colonne encoded e anno (TIME_PERIOD)
df_finale = pd.merge(
    df_raccolto, df_clima,
    how='left',
    left_on=['Territorio_encoded', 'TIME_PERIOD'],
    right_on=['Province', 'Year']
)

# Riempie i NaN numerici con la media (se ce ne sono dopo il merge)
df_finale.fillna(df_finale.mean(numeric_only=True), inplace=True)

# Salva il file finale
df_finale.to_csv("dataset_finale.csv", index=False)
