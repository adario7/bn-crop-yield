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
		
		# Create the base structure
		result_data = []
		
		# Group by year, territory, and crop
		groups = df_filtered.groupby(['TIME_PERIOD', 'Territorio', 'TYPE_OF_CROP'])
		
		print(f"Processing {len(groups)} unique combinations...", end="")
		
		processed_count = 0
		
		for (year, territorio, crop_code), group in groups:
			processed_count += 1
			
			# Show progress every 10,000 combinations
			if processed_count % 10000 == 0:
				print(".", end="", flush=True)
			
			# Initialize row
			row = {
				'year': year,
				'territorio': territorio,
				'crop': crop_code,  # Use TYPE_OF_CROP code instead of description
				'tot_surface': np.nan,
				'productive_surface': np.nan,
				'yield': np.nan,
				'yield_unit': None
			}
			
			# Extract values by indicator type
			indicators = {}
			for _, record in group.iterrows():
				data_type = record['DATA_TYPE']
				value = record['OBS_VALUE']
				if not pd.isna(value):
					indicators[data_type] = value
			
			# Calculate total surface (prefer ART, convert ART_ARE if needed)
			if 'ART' in indicators:
				row['tot_surface'] = indicators['ART']
			elif 'ART_ARE' in indicators:
				# Convert are to hectares (1 hectare = 100 are)
				row['tot_surface'] = indicators['ART_ARE'] / 100
			
			# Productive surface
			if 'PA_EXT' in indicators:
				row['productive_surface'] = indicators['PA_EXT']
			
			# Yield - prefer consistency within same crop
			# First check what yield types are available for this crop across all years
			if 'TP_HECT_EXT' in indicators:
				row['yield'] = indicators['TP_HECT_EXT']
				row['yield_unit'] = 'hectoliters'
			elif 'TP_QUIN_EXT' in indicators:
				row['yield'] = indicators['TP_QUIN_EXT']
				row['yield_unit'] = 'quintals'
			elif 'TP_THOQUIN_EXT' in indicators:
				# Convert thousands of quintals to quintals
				row['yield'] = indicators['TP_THOQUIN_EXT'] * 1000
				row['yield_unit'] = 'quintals'
			
			# Only add rows that have at least some data
			if not all(pd.isna([row['tot_surface'], row['productive_surface'], row['yield']])):
				result_data.append(row)
		
		# Create final dataframe
		result_df = pd.DataFrame(result_data)
		
		print()
		print(f"Created {len(result_df)} processed records")
		
		# Sort by year, territory, crop for better readability
		result_df = result_df.sort_values(['year', 'territorio', 'crop']).reset_index(drop=True)
		
		# Create final output (no yield_unit column)
		final_df = result_df[['year', 'territorio', 'crop', 'tot_surface', 'productive_surface', 'yield']].copy()
		
		# Create build directory if it doesn't exist
		os.makedirs('build', exist_ok=True)
		
		# Save to compressed CSV with semicolon separator
		final_df.to_csv(output_file, index=False, compression='gzip', sep=';')
		print(f"Saved processed data to: {output_file}")
		
		print(f"Final processing complete: {processed_count:,} combinations processed")
		
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

def analyze_yield_consistency():
	"""
	Analyze yield unit consistency across crops to ensure we're not mixing units
	"""
	print("\nAnalyzing yield unit consistency by crop...")
	
	try:
		df = pd.read_csv('data/crops.csv.gz', sep=';', compression='gzip', 
						usecols=range(23), engine='c')
		
		# Filter for yield indicators
		yield_types = ['TP_HECT_EXT', 'TP_QUIN_EXT', 'TP_THOQUIN_EXT']
		yield_df = df[df['DATA_TYPE'].isin(yield_types)]
		
		# Group by crop and see what yield types each crop uses
		crop_yield_types = yield_df.groupby('TYPE_OF_CROP')['DATA_TYPE'].unique()
		
		print("Sample yield consistency check (first 10 crops):")
		for i, (crop_code, yield_types_used) in enumerate(crop_yield_types.head(10).items()):
			yield_types_list = list(yield_types_used)
			print(f"  {crop_code}: {yield_types_list}")
			
	except Exception as e:
		print(f"Error in yield consistency analysis: {e}")

if __name__ == "__main__":
	# Process the data
	processed_df = process_crops_data()
	
	# Analyze yield consistency
	#analyze_yield_consistency()
	
	if processed_df is not None:
		print(f"\nProcessing completed successfully!")
		print(f"Output saved to: build/outputs.csv.gz")
