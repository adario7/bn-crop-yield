import pandas as pd

df_input = pd.read_csv("build/inputs.csv.gz", delimiter=';', compression='gzip')
df_output = pd.read_csv("build/outputs.csv.gz", delimiter=';', compression='gzip')

df_input.rename(columns={'Province': 'province', 'Year': 'year', 'Temperature_Mean_C': 'mean_temperature', 'Precipitation_Annual_mm': 'precipitation',
						 'ELEM_NUTR_HETT_KG_ANHY_FOSFOR': 'soil_anhy_fosfor',
						'ELEM_NUTR_HETT_KG_MCRNT_OX_POTAS': 'soil_mcrnt_ox_potas',
						'ELEM_NUTR_HETT_KG_NITROGEN': 'soil_nitrogen',
						'ELEM_NUTR_HETT_KG_ORG_COMP': 'soil_org_comp',
						 },
				inplace=True)
df_output.rename(columns={'territorio': 'province'}, inplace=True)

df = pd.merge(
    df_output, df_input,
    how='inner',
	on=['province', 'year']
)

print(f"Merged dataset shape: {df.shape}")
print("\nMissing values per column after merge:")
merged_missing = df.isnull().sum()
merged_total = len(df)
for col in df.columns:
    missing_count = merged_missing[col]
    missing_pct = (missing_count / merged_total) * 100
    print(f"  {col}: {missing_count:,} missing ({missing_pct:.2f}%)")

print(f"\nRows with at least one missing value: {df.isnull().any(axis=1).sum():,} ({(df.isnull().any(axis=1).sum() / merged_total) * 100:.2f}%)")

# Drop rows with missing values
df_clean = df.dropna()

df_clean.to_csv("build/dataset.csv.gz", index=False, compression='gzip')
