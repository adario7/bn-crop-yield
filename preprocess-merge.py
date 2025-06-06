import pandas as pd
import numpy as np

# Load the datasets
df_input = pd.read_csv("build/inputs.csv.gz", delimiter=';', compression='gzip')
df_output = pd.read_csv("build/outputs.csv.gz", delimiter=';', compression='gzip')

# Rename columns for consistency and clarity
df_input.rename(columns={'Province': 'province', 'Year': 'year', 'Temperature_Mean_C': 'mean_temperature', 'Precipitation_Annual_mm': 'precipitation',
						 'ELEM_NUTR_HETT_KG_ANHY_FOSFOR': 'soil_anhy_fosfor',
						'ELEM_NUTR_HETT_KG_MCRNT_OX_POTAS': 'soil_mcrnt_ox_potas',
						'ELEM_NUTR_HETT_KG_NITROGEN': 'soil_nitrogen',
						'ELEM_NUTR_HETT_KG_ORG_COMP': 'soil_org_comp',
						 },
				inplace=True)
df_output.rename(columns={'territorio': 'province'}, inplace=True)

# Merge the two dataframes
df = pd.merge(
    df_output, df_input,
    how='inner',
	on=['province', 'year']
)

print(f"Merged dataset shape: {df.shape}")

# --- New Imputation Logic ---
# Calculate the average productive_surface / tot_surface ratio from valid data
# A row is considered valid for this calculation if both values are present, tot_surface > 0,
# and productive_surface is not greater than tot_surface.
valid_ratio_rows = df[
    (df['productive_surface'].notna()) &
    (df['tot_surface'].notna()) &
    (df['tot_surface'] > 0) &
    # Ensure the ratio is physically possible
    (df['productive_surface'] <= df['tot_surface'])
].copy()

# Calculate the ratio for each valid row and then find the mean
if not valid_ratio_rows.empty:
    valid_ratio_rows['ps_ts_ratio'] = valid_ratio_rows['productive_surface'] / valid_ratio_rows['tot_surface']
    average_ratio = valid_ratio_rows['ps_ts_ratio'].mean()
else:
    average_ratio = 0 # Handle case with no valid rows to compute ratio from

# Identify rows that need imputation: 'productive_surface' is missing but 'tot_surface' exists.
rows_to_impute_mask = df['productive_surface'].isnull() & df['tot_surface'].notna() & (df['tot_surface'] > 0)
num_imputed = rows_to_impute_mask.sum()
total_rows = len(df)

# Report on the imputation
print("\n--- Productive Surface Imputation Report ---")
if average_ratio > 0 and num_imputed > 0:
    # Perform the imputation using the calculated average ratio
    df.loc[rows_to_impute_mask, 'productive_surface'] = df.loc[rows_to_impute_mask, 'tot_surface'] * average_ratio
    
    pct_imputed = (num_imputed / total_rows) * 100 if total_rows > 0 else 0
    print(f"  Average 'productive_surface / tot_surface' ratio: {average_ratio:.4f}")
    print(f"  Computed and filled {num_imputed:,} missing 'productive_surface' values ({pct_imputed:.2f}% of total rows).")
else:
    print("  No values were imputed for 'productive_surface'.")
print("------------------------------------------")


# --- Post-Imputation Analysis ---
print("\nMissing values per column after merge and imputation:")
merged_missing = df.isnull().sum()
for col in df.columns:
    missing_count = merged_missing[col]
    if total_rows > 0:
        missing_pct = (missing_count / total_rows) * 100
        print(f"  {col}: {missing_count:,} missing ({missing_pct:.2f}%)")
    else:
        print(f"  {col}: {missing_count:,} missing")

rows_with_any_missing = df.isnull().any(axis=1).sum()
pct_rows_with_missing = (rows_with_any_missing / total_rows) * 100 if total_rows > 0 else 0
print(f"\nRows with at least one missing value: {rows_with_any_missing:,} ({pct_rows_with_missing:.2f}%)")

# Drop any remaining rows with missing values
df_clean = df.dropna()

# Save the final cleaned dataset
df_clean.to_csv("build/dataset.csv.gz", index=False, compression='gzip')

print(f"\nFinal cleaned dataset shape: {df_clean.shape}")
print("Cleaned dataset saved to 'build/dataset.csv.gz'")
