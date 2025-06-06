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
		
		# Apply interpolation to fill missing data points
		print("Applying linear interpolation to fill missing data points...")
		final_df, interpolation_stats = interpolate_missing_data(final_df)
		
		# Create build directory if it doesn't exist
		os.makedirs('build', exist_ok=True)
		
		# Save to compressed CSV with semicolon separator, ensuring column order
		column_order = ['year', 'territorio', 'crop', 'tot_surface', 'productive_surface', 'yield']
		final_df[column_order].to_csv(output_file, index=False, compression='gzip', sep=';')
		print(f"Saved processed data to: {output_file}")
		
		print(f"Final processing complete: {len(final_df):,} records processed")
		
		# Show interpolation report
		print_interpolation_report(interpolation_stats)
		
		# Show detailed statistics
		print_detailed_stats(final_df, result_df)
		
		# Show sample of the data
		print("\nSample of processed data (first 10 rows):")
		print(final_df.head(10).to_string(index=False))
		
		return final_df
		
	except Exception as e:
		print(f"Error processing data: {e}")
		return None

def interpolate_missing_data(df):
	"""
	For each (territorio, crop) pair, interpolate missing values across years using linear interpolation.
	Returns the dataframe with interpolated values and statistics about interpolation.
	Optimized version using pandas groupby operations instead of loops.
	"""
	
	# Create a copy to avoid modifying the original
	df_interp = df.copy()
	
	# Track interpolation statistics
	interpolation_stats = {
		'tot_surface': {'original_missing': 0, 'interpolated': 0, 'total_values': 0},
		'productive_surface': {'original_missing': 0, 'interpolated': 0, 'total_values': 0},
		'yield': {'original_missing': 0, 'interpolated': 0, 'total_values': 0}
	}
	
	# Count original missing values
	data_columns = ['tot_surface', 'productive_surface', 'yield']
	for col in data_columns:
		interpolation_stats[col]['original_missing'] = df_interp[col].isna().sum()
		interpolation_stats[col]['total_values'] = len(df_interp)
	
	print(f"Processing interpolation using vectorized operations...")
	
	# Create a complete year range for each (territorio, crop) combination
	# First, get the min and max year for each group
	group_year_ranges = df_interp.groupby(['territorio', 'crop'])['year'].agg(['min', 'max']).reset_index()
	
	# Create complete year ranges for all groups
	complete_combinations = []
	for _, row in group_year_ranges.iterrows():
		years = list(range(row['min'], row['max'] + 1))
		for year in years:
			complete_combinations.append({
				'territorio': row['territorio'],
				'crop': row['crop'],
				'year': year
			})
	
	# Convert to DataFrame - this creates the complete time series for all groups
	complete_df = pd.DataFrame(complete_combinations)
	
	# Merge with original data to get the complete dataset with NaN gaps
	df_complete = complete_df.merge(
		df_interp, 
		on=['territorio', 'crop', 'year'], 
		how='left'
	)
	
	# Sort to ensure proper order for interpolation
	df_complete = df_complete.sort_values(['territorio', 'crop', 'year']).reset_index(drop=True)
	
	# Apply interpolation using groupby with transform
	# This applies interpolation within each (territorio, crop) group
	def interpolate_group(group):
		"""Apply linear interpolation to a group if it has enough non-NaN values"""
		result = group.copy()
		for col in data_columns:
			if result[col].notna().sum() >= 2:  # Need at least 2 points for interpolation
				result[col] = result[col].interpolate(method='linear')
		return result
	
	# Apply interpolation to each group
	df_interpolated = df_complete.groupby(['territorio', 'crop'], group_keys=False).apply(interpolate_group)
	
	# Reset index to ensure clean DataFrame
	df_interpolated = df_interpolated.reset_index(drop=True)
	
	# Remove rows that are still completely empty (all three columns are NaN)
	mask = (df_interpolated['tot_surface'].notna() | 
			df_interpolated['productive_surface'].notna() | 
			df_interpolated['yield'].notna())
	
	df_final = df_interpolated[mask].copy()
	
	# Update interpolation statistics based on final dataset
	for col in data_columns:
		interpolation_stats[col]['total_values'] = len(df_final)
		# Calculate interpolated count: final non-NA minus original non-NA
		final_non_na = df_final[col].notna().sum()
		original_non_na = df[col].notna().sum()
		interpolation_stats[col]['interpolated'] = max(0, final_non_na - original_non_na)
	
	return df_final, interpolation_stats

def print_interpolation_report(stats):
	"""Print a detailed report about interpolation statistics"""
	
	print(f"\nInterpolation Report:")
	print("=" * 50)
	
	for column, data in stats.items():
		total = data['total_values']
		interpolated = data['interpolated']
		
		if total > 0:
			interpolation_pct = (interpolated / total) * 100
			print(f"\n{column.replace('_', ' ').title()}:")
			print(f"  Total values in final dataset: {total:,}")
			print(f"  Values from interpolation: {interpolated:,}")
			print(f"  Percentage interpolated: {interpolation_pct:.2f}%")
		else:
			print(f"\n{column.replace('_', ' ').title()}: No data available")
	
	total_all_columns = sum(data['total_values'] for data in stats.values())
	total_interpolated = sum(data['interpolated'] for data in stats.values())
	
	if total_all_columns > 0:
		overall_pct = (total_interpolated / total_all_columns) * 100
		print(f"\nOverall Statistics:")
		print(f"  Total data points across all columns: {total_all_columns:,}")
		print(f"  Total interpolated values: {total_interpolated:,}")
		print(f"  Overall interpolation percentage: {overall_pct:.2f}%")

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
