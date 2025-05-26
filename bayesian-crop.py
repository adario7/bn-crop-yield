import pandas as pd
from pgmpy.estimators import HillClimbSearch, K2, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("build/dataset.csv.gz", compression="gzip")
print("Colonne:", df.columns.tolist())

# Discretizza la variabile target "Osservazione" (Raccolto) in 3 livelli
df["Raccolto_cat"] = pd.qcut(df["Osservazione"], q=3, labels=[0, 1, 2])
df["Raccolto_cat"] = df["Raccolto_cat"].astype("category")

# Discretizza anche temperatura e precipitazioni se non già categorizzate
df["Temp_cat"] = pd.qcut(df["Temperature_Mean_C"], q=3, labels=[0, 1, 2])
df["Pioggia_cat"] = pd.qcut(df["Precipitation_Annual_mm"], q=3, labels=[0, 1, 2])
df["Temp_cat"] = df["Temp_cat"].astype("category")
df["Pioggia_cat"] = df["Pioggia_cat"].astype("category")

# Discretizzazione dei nutrienti del terreno
nutrient_cols = {
    "Terreno_ELEM_NUTR_HETT_KG_ANHY_FOSFOR": "Terreno_Fosforo_cat",
    "Terreno_ELEM_NUTR_HETT_KG_NITROGEN": "Terreno_Azoto_cat",
    "Terreno_ELEM_NUTR_HETT_KG_ORG_COMP": "Terreno_Organico_cat",
    "Terreno_ELEM_NUTR_HETT_KG_MCRNT_OX_POTAS": "Terreno_Potassio_cat"
}

for original, cat_name in nutrient_cols.items():
    df[cat_name] = pd.qcut(df[original], q=3, labels=[0, 1, 2])
    df[cat_name] = df[cat_name].astype("category")

# Selezione delle variabili per la rete
features = [
    "TYPE_OF_CROP",
    "Temp_cat", "Pioggia_cat",
    "Terreno_Fosforo_cat", "Terreno_Azoto_cat",
    "Terreno_Organico_cat", "Terreno_Potassio_cat",
    "Raccolto_cat"
]

df_bn = df[features].copy()

# Structure learning
hc = HillClimbSearch(df_bn)
best_model = hc.estimate(scoring_method=K2(df_bn))

# Costruisci la rete
model = DiscreteBayesianNetwork(best_model.edges())
model.fit(df_bn, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)

# Inferenzia
infer = VariableElimination(model)
query = infer.query(
    variables=["Raccolto_cat"],
    evidence={
        "TYPE_OF_CROP": "MAIZE",
        "Temp_cat": 1,
        "Pioggia_cat": 2,
        "Terreno_Fosforo_cat": 2,
        "Terreno_Azoto_cat": 1,
        "Terreno_Organico_cat": 0,
        "Terreno_Potassio_cat": 1
    }
)
print(query)

for temp_cat in [0, 1, 2]:
    query = infer.query(
        variables=["Raccolto_cat"],
        evidence={
            "TYPE_OF_CROP": "MAIZE",
            "Temp_cat": temp_cat,
            "Pioggia_cat": 1,
            "Terreno_Fosforo_cat": 2,
            "Terreno_Azoto_cat": 1,
            "Terreno_Organico_cat": 0,
            "Terreno_Potassio_cat": 1
        }
    )
print(f"Temp_cat = {temp_cat} -> {query}")


pos = nx.shell_layout(model)  # alternativa più stabile a spring_layout

plt.figure(figsize=(12, 8))  # aumenta le dimensioni della figura
nx.draw_networkx_nodes(model, pos, node_size=3000, node_color='lightblue')
nx.draw_networkx_edges(model, pos, arrows=False)
nx.draw_networkx_labels(model, pos, font_size=10, font_weight='bold')
plt.axis('off')
plt.title("Rete Bayesiana crop")
plt.tight_layout()
plt.savefig("build/baesyan-network.png", dpi=300)  # salva il file per sicurezza
plt.show()
