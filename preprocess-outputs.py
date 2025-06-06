import pandas as pd
import numpy as np
import os

def process_crops_data(input_file='data/crops.csv.gz', output_file='build/outputs.csv.gz'):
	"""
	Process crops data to create a clean dataset with:
	year, territorio, crop, tot_surface, productive_surface, yield
	"""
	
	try:
		# Read the data
		print(f"Reading file: {input_file}")
		df = pd.read_csv(input_file, sep=';', compression='gzip', 
						usecols=range(23), engine='c')
		
		print(f"Successfully loaded {len(df)} rows")
		
		# Filter for relevant DATA_TYPE values
		relevant_types = ['ART', 'ART_ARE', 'PA_EXT', 'TP_HECT_EXT', 'TP_QUIN_EXT', 'TP_THOQUIN_EXT']
		df_filtered = df[df['DATA_TYPE'].isin(relevant_types)].copy()
		
		print(f"Filtered to {len(df_filtered)} rows with relevant indicators")
		
		# Convert numeric observation values, handle missing data
		df_filtered['OBS_VALUE'] = pd.to_numeric(df_filtered['Osservazione'], errors='coerce')
		
		# Clean territorio field - unescape quotes
		df_filtered['Territorio'] = df_filtered['Territorio'].str.replace("'L\"'Aquila'", "L'Aquila")
		df_filtered['Territorio'] = df_filtered['Territorio'].str.replace("'Reggio nell\"'Emilia'", 'Reggio nell\'Emilia')
		
		# Remove rows with NaN values in OBS_VALUE since they won't contribute to the final result
		df_filtered = df_filtered.dropna(subset=['OBS_VALUE'])
		
		print("Processing data using vectorized operations...")
		
		# Create pivot table to reshape data - each DATA_TYPE becomes a column
		pivot_df = df_filtered.pivot_table(
			index=['TIME_PERIOD', 'Territorio', 'TYPE_OF_CROP'],
			columns='DATA_TYPE',
			values='OBS_VALUE',
			aggfunc='first'  # In case of duplicates, take first value
		).reset_index()
		
		# Rename columns for clarity
		pivot_df.columns.name = None  # Remove the 'DATA_TYPE' name from columns
		pivot_df = pivot_df.rename(columns={
			'TIME_PERIOD': 'year',
			'Territorio': 'territorio', 
			'TYPE_OF_CROP': 'crop'
		})
		
		print(f"Created pivot table with {len(pivot_df)} records")
		
		# Calculate total surface (prefer ART, convert ART_ARE if needed)
		# Use vectorized operations with np.where for conditional logic
		pivot_df['tot_surface'] = np.where(
			pivot_df.get('ART', pd.Series()).notna(),
			pivot_df.get('ART', 0),
			np.where(
				pivot_df.get('ART_ARE', pd.Series()).notna(),
				pivot_df.get('ART_ARE', 0) / 100,  # Convert are to hectares
				np.nan
			)
		)
		
		# Productive surface - direct assignment
		pivot_df['productive_surface'] = pivot_df.get('PA_EXT', np.nan)
		
		# Yield calculation with priority: TP_HECT_EXT > TP_QUIN_EXT > TP_THOQUIN_EXT
		# Create yield column with cascading priorities
		pivot_df['yield'] = np.select([
			pivot_df.get('TP_HECT_EXT', pd.Series()).notna(),
			pivot_df.get('TP_QUIN_EXT', pd.Series()).notna(),
			pivot_df.get('TP_THOQUIN_EXT', pd.Series()).notna()
		], [
			pivot_df.get('TP_HECT_EXT', 0),
			pivot_df.get('TP_QUIN_EXT', 0), 
			pivot_df.get('TP_THOQUIN_EXT', 0) * 1000  # Convert thousands of quintals to quintals
		], default=np.nan)
		
		# Filter out rows that have no useful data (all three main columns are NaN)
		mask = (pivot_df['tot_surface'].notna() | 
				pivot_df['productive_surface'].notna() | 
				pivot_df['yield'].notna())
		
		result_df = pivot_df[mask].copy()
		
		print(f"Filtered to {len(result_df)} records with at least some data")
		
		# Sort by year, territory, crop for better readability  
		result_df = result_df.sort_values(['year', 'territorio', 'crop']).reset_index(drop=True)
		
		# Create final output with only the required columns
		final_df = result_df[['year', 'territorio', 'crop', 'tot_surface', 'productive_surface', 'yield']].copy()
		
		# Create build directory if it doesn't exist
		os.makedirs('build', exist_ok=True)
		
		# Save to compressed CSV with semicolon separator
		final_df.to_csv(output_file, index=False, compression='gzip', sep=';')
		print(f"Saved processed data to: {output_file}")
		
		print(f"Final processing complete: {len(final_df):,} records processed")
		
		# Show detailed statistics
		print_detailed_stats(final_df, result_df)
		
		# Show sample of the data
		print("\nSample of processed data (first 10 rows):")
		print(final_df.head(10).to_string(index=False))
		
		return final_df
		
	except Exception as e:
		print(f"Error processing data: {e}")
		return None

def print_detailed_stats(final_df, result_df):
	"""Print detailed statistics about the processed data and indicators used"""
	
	print(f"\nDetailed Statistics:")
	print("=" * 50)
	
	# Basic counts
	print(f"Total records processed: {len(final_df):,}")
	print(f"Years covered: {final_df['year'].min()} - {final_df['year'].max()}")
	print(f"Unique territories: {final_df['territorio'].nunique()}")
	print(f"Unique crops: {final_df['crop'].nunique()}")
	
	# Data source analysis
	print(f"\nData Source Analysis:")
	print("-" * 30)
	
	# Re-analyze the original data to get accurate counts
	try:
		df_orig = pd.read_csv('data/crops.csv.gz', sep=';', compression='gzip', 
							usecols=range(23), engine='c')
		relevant_types = ['ART', 'ART_ARE', 'PA_EXT', 'TP_HECT_EXT', 'TP_QUIN_EXT', 'TP_THOQUIN_EXT']
		df_filt = df_orig[df_orig['DATA_TYPE'].isin(relevant_types)]
		
		# Count crops using each surface indicator
		art_crops = set(df_filt[df_filt['DATA_TYPE'] == 'ART']['TYPE_OF_CROP'].unique())
		art_are_crops = set(df_filt[df_filt['DATA_TYPE'] == 'ART_ARE']['TYPE_OF_CROP'].unique())
		pa_crops = set(df_filt[df_filt['DATA_TYPE'] == 'PA_EXT']['TYPE_OF_CROP'].unique())
		
		# Count crops using each yield indicator
		hect_crops = set(df_filt[df_filt['DATA_TYPE'] == 'TP_HECT_EXT']['TYPE_OF_CROP'].unique())
		quin_crops = set(df_filt[df_filt['DATA_TYPE'] == 'TP_QUIN_EXT']['TYPE_OF_CROP'].unique())
		thoquin_crops = set(df_filt[df_filt['DATA_TYPE'] == 'TP_THOQUIN_EXT']['TYPE_OF_CROP'].unique())
		
		print(f"Surface Indicators Usage:")
		print(f"  ART (hectares): {len(art_crops)} crops")
		print(f"  ART_ARE (ares): {len(art_are_crops)} crops")
		print(f"  PA_EXT (productive surface): {len(pa_crops)} crops")
		
		print(f"\nYield Indicators Usage:")
		print(f"  TP_HECT_EXT (hectoliters): {len(hect_crops)} crops")
		print(f"  TP_QUIN_EXT (quintals): {len(quin_crops)} crops")
		print(f"  TP_THOQUIN_EXT (thousands of quintals): {len(thoquin_crops)} crops")
		
	except Exception as e:
		print(f"Error in detailed analysis: {e}")
	
	# Data completeness
	print(f"\nData Completeness:")
	print("-" * 20)
	total_records = len(final_df)
	print(f"Records with total surface: {final_df['tot_surface'].notna().sum():,} ({final_df['tot_surface'].notna().mean()*100:.1f}%)")
	print(f"Records with productive surface: {final_df['productive_surface'].notna().sum():,} ({final_df['productive_surface'].notna().mean()*100:.1f}%)")
	print(f"Records with yield data: {final_df['yield'].notna().sum():,} ({final_df['yield'].notna().mean()*100:.1f}%)")
	
	# Records per year
	print(f"\nRecords per year:")
	yearly_counts = final_df['year'].value_counts().sort_index()
	for year, count in yearly_counts.items():
		print(f"  {year}: {count:,} records")
	
	# Top territories by number of records
	print(f"\nTop 10 territories by number of records:")
	top_territories = final_df['territorio'].value_counts().head(10)
	for territory, count in top_territories.items():
		print(f"  {territory}: {count:,} records")

	# Top territories by number of records
	print(f"\nTop 10 crops by number of records:")
	top_crops = final_df['crop'].value_counts().head(10)
	for crop, count in top_crops.items():
		print(f"  {crop}: {count:,} records")

if __name__ == "__main__":
	# Process the data
	processed_df = process_crops_data()
	
	if processed_df is not None:
		print(f"\nProcessing completed successfully!")
		print(f"Output saved to: build/outputs.csv.gz")
