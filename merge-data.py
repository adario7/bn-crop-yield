import pandas as pd
import numpy as np
import os
import gzip
import re
from collections import defaultdict

# --- Configuration ---
DATA_DIR = 'data'
BUILD_DIR = 'build'
OUTPUT_FILE = os.path.join(BUILD_DIR, 'environment.csv.gz')

FILE_PIOGGIA = os.path.join(DATA_DIR, 'pioggia.csv.gz')
FILE_PRECIPITAZIONE = os.path.join(DATA_DIR, 'precipitazione-annua.csv.gz')
FILE_TEMP_MEDIA = os.path.join(DATA_DIR, 'temp-media.csv.gz')
FILE_TERRENO = os.path.join(DATA_DIR, 'terreno.csv.gz')
FILE_COMUNI = os.path.join(DATA_DIR, 'comuni-italiani.csv.gz')

DROP_ROW_NAN_RATIO_THRESHOLD = 0.9
DROP_COL_NAN_RATIO_THRESHOLD = 0.9
DROP_ROWS_IF_KEYS_MISSING = ['Province', 'Year']

# --- Helper Functions ---
def create_build_dir():
	if not os.path.exists(BUILD_DIR):
		os.makedirs(BUILD_DIR)
		print(f"Created directory: {BUILD_DIR}")

def clean_col_name(col_name):
	name = str(col_name)
	name = re.sub(r'\s+', '_', name)
	name = re.sub(r'[^\w\s-]', '', name)
	name = name.strip('_')
	return name if name else "unnamed_col"

def custom_number_converter(value_str):
	if pd.isna(value_str): return np.nan
	if isinstance(value_str, (int, float)): return float(value_str)
	s = str(value_str).strip()
	if not s: return np.nan
	num_dots = s.count('.')
	if num_dots > 1:
		parts = s.split('.')
		s = "".join(parts[:-1]) + "." + parts[-1]
	try:
		return float(s)
	except ValueError:
		return np.nan

def get_years_from_wide_format(file_path):
	years = set()
	try:
		df_header = pd.read_csv(file_path, sep=';', compression='gzip', nrows=0, encoding='utf-8')
	except Exception:
		try:
			df_header = pd.read_csv(file_path, sep=';', compression='gzip', nrows=0, encoding='latin1')
		except Exception as e:
			print(f"Could not read header for year scanning from {file_path}: {e}")
			return years
	for col in df_header.columns:
		if col.isdigit():
			years.add(int(col))
	return years

def get_years_from_long_format(file_path, year_column_name):
	years = set()
	try:
		df_year_col = pd.read_csv(file_path, sep=';', compression='gzip', usecols=[year_column_name], encoding='utf-8', dtype=str)
	except Exception:
		try:
			df_year_col = pd.read_csv(file_path, sep=';', compression='gzip', usecols=[year_column_name], encoding='latin1', dtype=str)
		except Exception as e:
			print(f"Could not read year column from {file_path}: {e}")
			return years
			
	numeric_years = pd.to_numeric(df_year_col[year_column_name], errors='coerce').dropna().unique()
	for yr in numeric_years:
		years.add(int(yr))
	return years

def load_comuni_mapping():
	print("Loading comuni mapping data...")
	try:
		comuni_df = pd.read_csv(FILE_COMUNI, sep=';', compression='gzip', encoding='utf-8', dtype=str)
	except Exception:
		comuni_df = pd.read_csv(FILE_COMUNI, sep=';', compression='gzip', encoding='latin1', dtype=str)

	df1 = comuni_df[['Denominazione provincia', 'Codice NUTS2 2006 (3)']].copy()
	df1.rename(columns={'Denominazione provincia': 'Province', 'Codice NUTS2 2006 (3)': 'NUTS2_Code'}, inplace=True)
	df1.dropna(subset=['Province', 'NUTS2_Code'], inplace=True)
	df1 = df1[df1['Province'] != '-'] # Exclude "-" province
	df1 = df1.drop_duplicates()

	df2 = comuni_df[['Denominazione Città metropolitana', 'Codice NUTS2 2006 (3)']].copy()
	df2.rename(columns={'Denominazione Città metropolitana': 'Province', 'Codice NUTS2 2006 (3)': 'NUTS2_Code'}, inplace=True)
	df2.dropna(subset=['Province', 'NUTS2_Code'], inplace=True)
	df2 = df2[df2['Province'] != '-'] # Exclude "-" province
	df2 = df2.drop_duplicates()

	df = pd.concat([df1, df2], ignore_index=True)
	# provincie mulitple come Barletta-Andria-Trani
	df['City'] = df['Province'].str.split('-').str[0]
	df['Province'] = df['Province'].str.replace('/', ' / ')

	df.loc[df['Province'] == "Valle d'Aosta / Vallée d'Aoste", 'City'] = 'Aosta'
	df.loc[df['Province'] == 'Bolzano / Bozen', 'City'] = 'Bolzano'

	return df

def interpolate_and_reindex_df(df, id_cols, value_cols, global_min_year, global_max_year, interpolation_stats_agg, col_prefix=""):
	"""Helper to reindex, interpolate, and track stats for a processed DataFrame."""
	if df.empty:
		return pd.DataFrame(columns=id_cols + value_cols), interpolation_stats_agg

	# Ensure 'Year' is integer for proper indexing
	df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
	df.dropna(subset=['Year'] + [id_col for id_col in id_cols if id_col != 'Year'], inplace=True) # Drop if Year or main ID is NaN

	# Exclude province named "-"
	if 'Province' in df.columns:
		df = df[df['Province'] != '-']
		if df.empty:
			return pd.DataFrame(columns=id_cols + value_cols), interpolation_stats_agg


	reindexed_dfs = []
	all_years_pd_index = pd.RangeIndex(start=global_min_year, stop=global_max_year + 1, name='Year')
	
	# Group by all ID columns except 'Year' (typically just 'Province')
	grouping_cols = [col for col in id_cols if col != 'Year']

	if not grouping_cols : # Should not happen if 'Province' is an id_col
		print(f"Warning: No grouping columns for interpolation for prefix {col_prefix}. Skipping reindex/interpolation for this df.")
		# Apply prefix to value columns if any exist
		rename_map = {vc: f"{col_prefix}{clean_col_name(vc)}" for vc in value_cols}
		df.rename(columns=rename_map, inplace=True)
		return df, interpolation_stats_agg


	for group_keys, group_df in df.groupby(grouping_cols):
		prov_df = group_df.set_index('Year')
		prov_df = prov_df[~prov_df.index.duplicated(keep='first')] 
		
		prov_df_reindexed = prov_df.reindex(all_years_pd_index)
		
		# Restore grouping keys (Province name)
		if isinstance(group_keys, tuple):
			for i, key_col in enumerate(grouping_cols):
				prov_df_reindexed[key_col] = group_keys[i]
		else: # Single grouping column
			prov_df_reindexed[grouping_cols[0]] = group_keys
		
		current_value_cols_in_df = [vc for vc in value_cols if vc in prov_df_reindexed.columns]

		for v_col in current_value_cols_in_df:
			is_na_before = prov_df_reindexed[v_col].isna()
			prov_df_reindexed[v_col] = prov_df_reindexed[v_col].interpolate(method='linear', axis=0, limit_direction='both')
			interpolated_in_col = (is_na_before & (~prov_df_reindexed[v_col].isna())).sum()
			
			final_col_name = f"{col_prefix}{clean_col_name(v_col)}"
			interpolation_stats_agg[final_col_name] = interpolation_stats_agg.get(final_col_name, 0) + interpolated_in_col
			
		reindexed_dfs.append(prov_df_reindexed.reset_index())

	if not reindexed_dfs:
		# Construct empty DF with correctly prefixed columns
		final_value_cols_prefixed = [f"{col_prefix}{clean_col_name(vc)}" for vc in value_cols]
		return pd.DataFrame(columns=id_cols + final_value_cols_prefixed), interpolation_stats_agg
	
	processed_df = pd.concat(reindexed_dfs, ignore_index=True)
	
	# Apply prefix to value columns in the concatenated DataFrame
	rename_map = {vc: f"{col_prefix}{clean_col_name(vc)}" for vc in value_cols if vc in processed_df.columns}
	processed_df.rename(columns=rename_map, inplace=True)
	
	return processed_df, interpolation_stats_agg

# --- Data Processing Functions ---
def process_pioggia(province_map, global_min_year, global_max_year, interpolation_stats_agg):
	print("Processing pioggia data...")
	try:
		df = pd.read_csv(FILE_PIOGGIA, sep=';', compression='gzip', encoding='utf-8', low_memory=False)
	except Exception:
		df = pd.read_csv(FILE_PIOGGIA, sep=';', compression='gzip', encoding='latin1', low_memory=False)

	df = df[['Territorio', 'DATA_TYPE', 'TIME_PERIOD', 'Osservazione']].copy()
	df.rename(columns={'TIME_PERIOD': 'Year', 'Osservazione': 'Value', 'DATA_TYPE': 'Indicator_Code', 'Territorio': 'Province'}, inplace=True)

	df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
	df.dropna(subset=['Year', 'Value', 'Province', 'Indicator_Code'], inplace=True)
	
	df = df[df['Province'].isin(province_map['Province'])]

	try:
		pioggia_pivot = df.pivot_table(index=['Province', 'Year'], columns='Indicator_Code', values='Value').reset_index()
	except Exception:
		df_agg = df.groupby(['Province', 'Year', 'Indicator_Code'])['Value'].mean().reset_index()
		pioggia_pivot = df_agg.pivot_table(index=['Province', 'Year'], columns='Indicator_Code', values='Value').reset_index()

	id_cols = ['Province', 'Year']
	value_cols = [col for col in pioggia_pivot.columns if col not in id_cols]
	
	pioggia_final, interpolation_stats_agg = interpolate_and_reindex_df(
		pioggia_pivot, id_cols, value_cols, global_min_year, global_max_year, interpolation_stats_agg, "Pioggia_"
	)
	print(f"Pioggia data processed: {pioggia_final.shape}")
	return pioggia_final, interpolation_stats_agg

def process_precipitazione(province_region_map, global_min_year, global_max_year, interpolation_stats_agg):
	print("Processing precipitazione-annua data...")
	try:
		header_df = pd.read_csv(FILE_PRECIPITAZIONE, sep=';', compression='gzip', encoding='utf-8', nrows=0)
	except Exception:
		header_df = pd.read_csv(FILE_PRECIPITAZIONE, sep=';', compression='gzip', encoding='latin1', nrows=0)
	
	year_columns = [col for col in header_df.columns if col.isdigit()]
	converters = {year_col: custom_number_converter for year_col in year_columns}
	
	try:
		df = pd.read_csv(FILE_PRECIPITAZIONE, sep=';', compression='gzip', encoding='utf-8', converters=converters)
	except Exception:
		df = pd.read_csv(FILE_PRECIPITAZIONE, sep=';', compression='gzip', encoding='latin1', converters=converters)

	df.rename(columns={'Comune': 'City'}, inplace=True)
	df = df.merge(province_region_map, on='City', how='inner')
	df = df.drop(columns=['City', 'NUTS2_Code'])
	df = df.groupby('Province', as_index=False).mean()
	
	df_melted = df.melt(id_vars=['Province'], value_vars=year_columns, var_name='Year', value_name='Precipitation_Annual_mm')
	df_melted['Precipitation_Annual_mm'] = pd.to_numeric(df_melted['Precipitation_Annual_mm'], errors='coerce')

	id_cols = ['Province', 'Year']
	value_cols = ['Precipitation_Annual_mm']

	precip_final, interpolation_stats_agg = interpolate_and_reindex_df(
		df_melted, id_cols, value_cols, global_min_year, global_max_year, interpolation_stats_agg, "" # No prefix needed as value_name is unique
	)
	print(f"Precipitazione-annua data processed: {precip_final.shape}")
	return precip_final, interpolation_stats_agg

def process_temp_media(province_region_map, global_min_year, global_max_year, interpolation_stats_agg):
	print("Processing temp-media data...")
	try:
		df = pd.read_csv(FILE_TEMP_MEDIA, sep=';', compression='gzip', encoding='utf-8', decimal='.')
	except Exception:
		df = pd.read_csv(FILE_TEMP_MEDIA, sep=';', compression='gzip', encoding='latin1', decimal='.')

	df.rename(columns={'Comune': 'City'}, inplace=True)
	df = df.merge(province_region_map, on='City', how='inner')
	df = df.drop(columns=['City', 'NUTS2_Code'])
	df = df.groupby('Province', as_index=False).mean()

	year_columns = [col for col in df.columns if col.isdigit()]
	
	df_melted = df.melt(id_vars=['Province'], value_vars=year_columns, var_name='Year', value_name='Temperature_Mean_C')
	df_melted['Temperature_Mean_C'] = pd.to_numeric(df_melted['Temperature_Mean_C'], errors='coerce')

	id_cols = ['Province', 'Year']
	value_cols = ['Temperature_Mean_C']
	
	temp_final, interpolation_stats_agg = interpolate_and_reindex_df(
		df_melted, id_cols, value_cols, global_min_year, global_max_year, interpolation_stats_agg, "" # No prefix
	)
	print(f"Temp-media data processed: {temp_final.shape}")
	return temp_final, interpolation_stats_agg

def process_terreno(province_region_map, global_min_year, global_max_year, interpolation_stats_agg):
	print("Processing terreno data...")
	try:
		df = pd.read_csv(FILE_TERRENO, sep=';', compression='gzip', encoding='utf-8', low_memory=False)
	except Exception:
		df = pd.read_csv(FILE_TERRENO, sep=';', compression='gzip', encoding='latin1', low_memory=False)

	df = df[['REF_AREA', 'DATA_TYPE', 'TYPE_OF_PLANT_NUTRIENTS', 'TIME_PERIOD', 'Osservazione']].copy()
	df.rename(columns={'REF_AREA': 'NUTS2_Code', 'TIME_PERIOD': 'Year', 'Osservazione': 'Value'}, inplace=True)

	df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
	df.dropna(subset=['NUTS2_Code', 'DATA_TYPE', 'TYPE_OF_PLANT_NUTRIENTS', 'Year', 'Value'], inplace=True)

	df['Indicator_Terreno'] = df['DATA_TYPE'] + '_' + df['TYPE_OF_PLANT_NUTRIENTS']
	
	try:
		terreno_pivot = df.pivot_table(index=['NUTS2_Code', 'Year'], columns='Indicator_Terreno', values='Value').reset_index()
	except Exception:
		df_agg = df.groupby(['NUTS2_Code', 'Year', 'Indicator_Terreno'])['Value'].mean().reset_index()
		terreno_pivot = df_agg.pivot_table(index=['NUTS2_Code', 'Year'], columns='Indicator_Terreno', values='Value').reset_index()
	
	terreno_mapped = pd.merge(terreno_pivot, province_region_map, on='NUTS2_Code', how='left')
	terreno_mapped = terreno_mapped.drop(columns='City')
	#print("TERRENO MAPPED 1")
	#print(terreno_mapped.sample(10))
	terreno_mapped.dropna(subset=['Province'], inplace=True)
	terreno_mapped = terreno_mapped[terreno_mapped['Province'] != '-']
	#print("TERRENO MAPPED 2")
	#print(terreno_mapped.sample(10))
	
	id_cols = ['Province', 'Year']
	value_cols = [col for col in terreno_mapped.columns if col not in id_cols + ['NUTS2_Code']]
	# Select only relevant columns before passing to interpolate_and_reindex_df
	terreno_mapped_subset = terreno_mapped[id_cols + value_cols]

	terreno_final, interpolation_stats_agg = interpolate_and_reindex_df(
		terreno_mapped_subset, id_cols, value_cols, global_min_year, global_max_year, interpolation_stats_agg, "Terreno_"
	)
	print(f"Terreno data processed: {terreno_final.shape}")
	return terreno_final, interpolation_stats_agg

# --- Main Script Logic ---
def main():
	create_build_dir()
	
	print("Determining global year range...")
	all_years_collected = set()
	all_years_collected.update(get_years_from_wide_format(FILE_PRECIPITAZIONE))
	all_years_collected.update(get_years_from_wide_format(FILE_TEMP_MEDIA))
	all_years_collected.update(get_years_from_long_format(FILE_PIOGGIA, 'TIME_PERIOD'))
	all_years_collected.update(get_years_from_long_format(FILE_TERRENO, 'TIME_PERIOD'))

	if not all_years_collected:
		print("Critical: No years found in data. Cannot proceed with defined year range. Exiting.")
		return
	
	global_min_year = min(all_years_collected)
	global_max_year = max(all_years_collected)
	print(f"Global year range determined: {global_min_year} - {global_max_year}")

	province_region_map = load_comuni_mapping()
	
	interpolation_stats_agg = defaultdict(int) # Use defaultdict for convenience

	df_temp_media, interpolation_stats_agg = process_temp_media(province_region_map, global_min_year, global_max_year, interpolation_stats_agg)
	df_precipitazione, interpolation_stats_agg = process_precipitazione(province_region_map, global_min_year, global_max_year, interpolation_stats_agg)
	df_terreno, interpolation_stats_agg = process_terreno(province_region_map, global_min_year, global_max_year, interpolation_stats_agg)
	df_pioggia, interpolation_stats_agg = process_pioggia(province_region_map, global_min_year, global_max_year, interpolation_stats_agg)
	
	print("Merging datasets...")
	# Start with a base of all provinces and all years to ensure full coverage
	all_provinces_in_scope = set(df_temp_media['Province'].unique()) \
		.union(set(df_precipitazione['Province'].unique())) \
		.union(set(df_terreno['Province'].unique())) \
		.union(set(df_pioggia['Province'].unique()))
	
	# Filter out "-" if it somehow slipped through (should be handled in processing funcs)
	all_provinces_in_scope = {p for p in all_provinces_in_scope if p != '-'}


	if not all_provinces_in_scope:
		print("No valid provinces found after processing all datasets. Merged file will be empty or contain only headers.")
		# Create an empty DataFrame with expected key columns or specific structure if needed.
		merged_df = pd.DataFrame(columns=['Province', 'Year']) 
	else:
		year_range = pd.RangeIndex(start=global_min_year, stop=global_max_year + 1, name='Year')
		merged_df = pd.MultiIndex.from_product([list(all_provinces_in_scope), year_range], names=['Province', 'Year']).to_frame(index=False)
		
		datasets_to_merge = [df_temp_media, df_precipitazione, df_terreno, df_pioggia]
		for i, df_to_merge in enumerate(datasets_to_merge):
			if not df_to_merge.empty:
				# Ensure 'Year' is of a compatible type for merging, if it became object
				if 'Year' in df_to_merge.columns and df_to_merge['Year'].dtype == object:
					 df_to_merge['Year'] = pd.to_numeric(df_to_merge['Year'], errors='coerce')
				
				# Check for empty df_to_merge again after potential changes
				if not df_to_merge.empty and 'Province' in df_to_merge.columns and 'Year' in df_to_merge.columns:
					 merged_df = pd.merge(merged_df, df_to_merge, on=['Province', 'Year'], how='outer')
					 print(f"Shape after merging dataset {i+1}: {merged_df.shape}")
				else:
					print(f"Skipping merge for dataset {i+1} as it's empty or missing key columns after processing.")
			else:
				 print(f"Skipping merge for dataset {i+1} as it's empty before merge attempt.")


	print("Handling missing data post-merge...")
	cols_before_dropping = set(merged_df.columns)
	if DROP_ROWS_IF_KEYS_MISSING:
		initial_rows = len(merged_df)
		merged_df.dropna(subset=DROP_ROWS_IF_KEYS_MISSING, how='any', inplace=True)
		print(f"Dropped {initial_rows - len(merged_df)} rows due to missing keys: {DROP_ROWS_IF_KEYS_MISSING}")

	if DROP_ROW_NAN_RATIO_THRESHOLD is not None and not merged_df.empty:
		initial_rows = len(merged_df)
		min_non_na_values_row = int(len(merged_df.columns) * (1 - DROP_ROW_NAN_RATIO_THRESHOLD))
		merged_df.dropna(axis=0, thresh=min_non_na_values_row, inplace=True)
		print(f"Dropped {initial_rows - len(merged_df)} rows with >{DROP_ROW_NAN_RATIO_THRESHOLD*100}% NaNs.")

	if DROP_COL_NAN_RATIO_THRESHOLD is not None and not merged_df.empty:
		initial_cols = len(merged_df.columns)
		min_non_na_values_col = int(len(merged_df) * (1 - DROP_COL_NAN_RATIO_THRESHOLD))
		merged_df.dropna(axis=1, thresh=min_non_na_values_col, inplace=True)
		print(f"Dropped {initial_cols - len(merged_df.columns)} columns with >{DROP_COL_NAN_RATIO_THRESHOLD*100}% NaNs.")
	
	cols_after_dropping = set(merged_df.columns)
	dropped_cols_list = list(cols_before_dropping - cols_after_dropping)

	if 'Year' in merged_df.columns:
		 merged_df['Year'] = pd.to_numeric(merged_df['Year'], errors='coerce').astype('Int64')
	if 'Province' in merged_df.columns and 'Year' in merged_df.columns:
		merged_df.sort_values(by=['Province', 'Year'], inplace=True)
	
	print(f"Saving merged data to {OUTPUT_FILE}...")
	merged_df.to_csv(OUTPUT_FILE, sep=';', index=False, compression='gzip', encoding='utf-8')
	
	print("\n--- Script Finished ---")
	print(f"Final merged dataset shape: {merged_df.shape}")
	
	print("\n--- Dropped Columns Statistics ---")
	if dropped_cols_list:
		print("The following columns were dropped due to high NaN ratio:")
		for col in dropped_cols_list:
			print(f"- {col}")
	else:
		print("No columns were dropped based on the NaN threshold.")

	print("\n--- Interpolation Statistics ---")
	if not merged_df.empty:
		rows = []
		for col in merged_df.columns:
			if col in ['Province', 'Year']:
				continue

			total_final_non_na = merged_df[col].count()
			interpolated_count = interpolation_stats_agg.get(col, 0)

			if total_final_non_na > 0 and interpolated_count > 0:
				percent = (interpolated_count / total_final_non_na) * 100
				rows.append((col, total_final_non_na, interpolated_count, f"{percent:.2f}%"))
			elif interpolated_count > 0:
				rows.append((col, 0, interpolated_count, "All NaN"))

		summary_df = pd.DataFrame(rows, columns=['Column', 'Non-NA Final Count', 'Interpolated Count', 'Interpolated %'])
		print(summary_df.to_string(index=False))
	else:
		print("Merged DataFrame is empty, no interpolation statistics to report.")


	print("\nSample of the final data (first 5 rows):")
	print(merged_df.sample(5))

if __name__ == '__main__':
	main()
