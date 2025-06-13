import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, ExpertKnowledge
from pgmpy.inference import VariableElimination

# Configuration
N_QUANTILES = 3
MAX_INDEGREE = 6
TRY_ALL_CROPS = True
SEL_CROP = "WHEATD"  # ignored if TRY_ALL_CROPS = true
RATIO = 'tot_surface' # 'productive_surface' # None

DATASET_PATH = 'build/dataset.csv.gz'
OUTPUT_DIR = 'build/'
NETWORK_PLOT_PATH = os.path.join(OUTPUT_DIR, 'bayesian-network.png')
CROP_ACC_PATH = os.path.join(OUTPUT_DIR, 'accuracy-by-crop.csv')

# Define independent and dependent variables
VAR_TEMP = [ 'mean_temperature', 'precipitation', 'heat_days', 'heavy_rain_days', 'dry_days' ]
VAR_SOIL = [ 'soil_anhy_fosfor', 'soil_mcrnt_ox_potas', 'soil_nitrogen', 'soil_org_comp' ]
INDEPENDENT_VARS = VAR_TEMP + VAR_SOIL
DEPENDENT_VAR_NAME = 'yield_ratio'

FORBIDDEN_EDGES = [ (a, b) for a in VAR_TEMP for b in VAR_SOIL ] + [ (b, a) for a in VAR_TEMP for b in VAR_SOIL ] + [ (a, b) for a in VAR_SOIL for b in VAR_SOIL if a != b ] + [ (DEPENDENT_VAR_NAME, a) for a in INDEPENDENT_VARS ]

INFERENCE_CONDITION =  [] # VAR_SOIL

def ensure_output_dir():
	"""Ensures the output directory exists."""
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	print(f"Output directory '{OUTPUT_DIR}' ensured.")

def get_all_crops_available(filepath):
	print(f"Getting the available crops from '{filepath}'...")
	try:
		with gzip.open(filepath, 'rt') as f: return pd.read_csv(f)['crop'].unique()
	except FileNotFoundError:
		print(f"ERROR: Dataset not found at {filepath}. Please ensure it exists.")
		return []

def load_and_preprocess_data(filepath, num_quantiles_config):
	"""
	Loads data, selects the most frequent crop, calculates yield_ratio,
	and discretizes variables into quantiles.
	Uses num_quantiles_config as the target number of quantiles.
	"""
	print(f"Loading data from '{filepath}'...")
	try:
		with gzip.open(filepath, 'rt') as f:
			df = pd.read_csv(f)
	except FileNotFoundError:
		print(f"ERROR: Dataset not found at {filepath}. Please ensure it exists.")
		return None, None, None, None

	print(f"Initial dataset shape: {df.shape}")

	# Find the crop with the most rowsAdd commentMore actions
	print(df['crop'].value_counts().head(10))
	if isinstance(SEL_CROP, str):
		most_frequent_crop = SEL_CROP
	elif SEL_CROP is True:
		most_frequent_crop = df['crop'].mode()[0]
	else:
		most_frequent_crop = None
	print(f"Most frequent crop: '{most_frequent_crop}'")
	if most_frequent_crop is not None:
		df_crop = df[df['crop'] == most_frequent_crop].copy()
	else:
		df_crop = df
	print(f"Shape after filtering for crop '{most_frequent_crop}': {df_crop.shape}")

	# Calculate yield_ratio
	df_crop.loc[:, 'productive_surface'] = df_crop['productive_surface'].replace(0, np.nan) 
	if RATIO is not None:
		df_crop.loc[:, DEPENDENT_VAR_NAME] = df_crop['yield'] / df_crop[RATIO]
	else:
		df_crop.loc[:, DEPENDENT_VAR_NAME] = df_crop['yield']
	
	df_crop.replace([np.inf, -np.inf], np.nan, inplace=True)
	
	all_vars_for_analysis = INDEPENDENT_VARS + [DEPENDENT_VAR_NAME]
	# Crucial: Drop NaNs only from the subset of columns we will actually use for analysis *before* discretization
	df_crop.dropna(subset=all_vars_for_analysis, inplace=True)
	print(f"Shape after dropping NaNs in relevant columns ({len(all_vars_for_analysis)} cols): {df_crop.shape}")

	if df_crop.empty:
		print("ERROR: No data remaining after preprocessing (NaN drop). Cannot proceed.")
		return None, None, None, None

	print("Data sample:")
	print(df_crop.sample(n=min(5, len(df_crop))))

	discretized_df = pd.DataFrame()
	# Generate base quantile labels, e.g., ['q0', 'q1', ..., 'q9'] for num_quantiles_config=10
	base_quantile_labels = [f'q{i}' for i in range(num_quantiles_config)]
	
	original_continuous_data = df_crop[all_vars_for_analysis].copy()

	for col in all_vars_for_analysis:
		print(f"Discretizing column '{col}'...")
		if df_crop[col].isnull().any():
			# This should not happen if dropna was effective on all_vars_for_analysis
			print(f"ERROR: Column '{col}' still contains NaNs before discretization. This is unexpected.")
			return None, None, None, None

		try:
			# Attempt to discretize using quantiles
			discretized_df[col] = pd.qcut(df_crop[col], q=num_quantiles_config, labels=base_quantile_labels, duplicates='drop')
			# Check actual number of bins created by qcut
			actual_bins_created = discretized_df[col].nunique()
			print(f"  Successfully discretized '{col}' using qcut. Target quantiles: {num_quantiles_config}, Actual bins: {actual_bins_created}.")
		except ValueError as e:
			print(f"  Warning: qcut failed for column '{col}' with target {num_quantiles_config} quantiles: {e}.")
			n_unique = df_crop[col].nunique()
			
			if n_unique == 0: # Should be caught by earlier df_crop.empty check if all columns are empty
				print(f"  ERROR: Column '{col}' has no unique values after NaNs were dropped. Cannot discretize.")
				return None, None, None, None
			
			# Fallback: Use pd.cut. Determine the number of bins for pd.cut.
			# It should be at most num_quantiles_config, and at most n_unique.
			# If n_unique is 1, we create 1 bin.
			target_bins_for_cut = min(num_quantiles_config, n_unique)
			
			print(f"  Fallback for '{col}': Attempting pd.cut. n_unique={n_unique}, target_bins_for_cut={target_bins_for_cut}.")

			if target_bins_for_cut == 1:
				# All data falls into a single category.
				# Create a single label for this single bin.
				current_labels = [base_quantile_labels[0]] # Use 'q0'
				discretized_df[col] = pd.cut(df_crop[col], bins=1, labels=current_labels, include_lowest=True)
				print(f"	Used pd.cut with 1 bin for '{col}'.")
			elif target_bins_for_cut > 1:
				# Create labels for these bins. We need target_bins_for_cut labels.
				current_labels = base_quantile_labels[:target_bins_for_cut]
				try:
					discretized_df[col] = pd.cut(df_crop[col], bins=target_bins_for_cut, labels=current_labels, include_lowest=True, duplicates='drop')
					actual_bins_created = discretized_df[col].nunique()
					print(f"	Used pd.cut with target {target_bins_for_cut} bins for '{col}'. Actual bins: {actual_bins_created}.")
					if actual_bins_created < target_bins_for_cut:
						print(f"	Note: pd.cut created fewer bins ({actual_bins_created}) than targeted ({target_bins_for_cut}) for '{col}' due to data distribution and 'duplicates=drop'.")
				except ValueError as e_cut:
					print(f"  ERROR: pd.cut also failed for column '{col}' with {target_bins_for_cut} bins: {e_cut}.")
					print(f"	Unique values in '{col}' (count: {n_unique}): {np.sort(df_crop[col].unique().tolist())}")
					return None, None, None, None # Critical error if both qcut and cut fail
			else: # target_bins_for_cut is 0 (or less, which is impossible if n_unique >= 0)
				print(f"  ERROR: Unexpected state for column '{col}'. n_unique={n_unique}, target_bins_for_cut={target_bins_for_cut}.")
				return None, None, None, None
		
		# Final check for NaNs in discretized column, which can happen if pd.cut/qcut fails to assign some values
		if discretized_df[col].isnull().any():
			print(f"  ERROR: Column '{col}' has NaNs after discretization. This indicates a problem with binning all values.")
			print(f"	Original values that are NaN after discretization for '{col}':")
			print(df_crop[col][discretized_df[col].isnull()].unique())
			return None, None, None, None


	print(f"\nDiscretized data shape: {discretized_df.shape}")
	print("Sample of discretized data:")
	print(discretized_df.head())
	
	# Verify no NaNs in the final discretized_df for the columns used
	if discretized_df[all_vars_for_analysis].isnull().any().any():
		print("\nERROR: NaNs found in discretized data. This should not happen.")
		print(discretized_df.isnull().sum())
		return None, None, None, None

	return discretized_df


def discover_causal_structure_pc(data, independent_vars, dependent_var):
	"""
	Discovers causal structure using the PC algorithm.
	"""
	print("\nDiscovering causal structure using PC algorithm...")
	all_vars = independent_vars + [dependent_var]
	
	# Ensure data for PC only contains the relevant columns
	data_for_pc = data[all_vars].copy()

	try:
		# scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g, ll-cg, aic-cg, bic-cg
		hc_estimator = HillClimbSearch(data=data_for_pc)
		model_dag = hc_estimator.estimate(tabu_length=1000, max_indegree=MAX_INDEGREE, expert_knowledge=ExpertKnowledge(forbidden_edges=FORBIDDEN_EDGES), scoring_method="k2")
		print("HillClimbSearch with bicscore completed.")
	except Exception as e_hc:
		print(f"HillClimbSearch failed: {e_hc}")
		print("ERROR: Could not determine causal structure. Aborting.")
		return None

	print("model dag is a " + str(model_dag.__class__))
	if not model_dag.edges():
		print("Warning: Structure learning algorithm did not find any causal edges. The network will be disconnected.")
		# Create a DAG with all nodes but no edges if PC returns an empty graph object
		# but ensure it's a DAG, not None or other type.
		if not isinstance(model_dag, BayesianNetwork): # PC might return a list of edges or a Graph object
			temp_dag = BayesianNetwork()
			temp_dag.add_nodes_from(all_vars)
			if hasattr(model_dag, 'edges'): # If it's a graph-like object with edges
				temp_dag.add_edges_from(model_dag.edges())
			model_dag = temp_dag # Ensure it's a BayesianNetwork (DAG) instance
		elif not model_dag.nodes(): # If it's a BayesianNetwork but has no nodes (empty graph)
			model_dag.add_nodes_from(all_vars)


	else:
		print("Discovered causal edges:")
		for edge in model_dag.edges():
			print(f"  {edge[0]} -> {edge[1]}")
	return model_dag


def train_bayesian_network(data, structure_dag, independent_vars, dependent_var):
	"""
	Trains a Bayesian Network on the discovered graph structure.
	structure_dag is expected to be a pgmpy.models.BayesianNetwork (DAG) instance.
	"""
	print("\nTraining Bayesian Network...")
	all_vars = independent_vars + [dependent_var]
	
	# The structure_dag from PC or HillClimb should already be a DAG (BayesianNetwork instance)
	# containing nodes and edges.
	bn_model = DiscreteBayesianNetwork() 
	bn_model.add_nodes_from(structure_dag.nodes())
	bn_model.add_edges_from(structure_dag.edges())

	# Ensure all nodes in the model are in the data columns for fitting
	nodes_for_fitting = list(bn_model.nodes())
	missing_nodes_in_data = [node for node in nodes_for_fitting if node not in data.columns]
	if missing_nodes_in_data:
		print(f"ERROR: Nodes {missing_nodes_in_data} are in the model structure but not in the data columns for fitting.")
		return None

	# Filter data to only include nodes present in the model, in the correct order for pgmpy
	data_for_fitting = data[nodes_for_fitting]

	try:
		bn_model.fit(data_for_fitting, estimator=MaximumLikelihoodEstimator)
		print("Bayesian Network training complete using MaximumLikelihoodEstimator.")
		
		if bn_model.check_model():
			print("Model is consistent (all CPDs sum to 1, variables match).")
		else:
			print("Warning: Model is not consistent after fitting. This is unexpected.")
			# Detailed check for CPDs if inconsistent
			for node_name in bn_model.nodes():
				try:
					cpd = bn_model.get_cpds(node_name)
					if cpd is None:
						print(f"  CPD for node '{node_name}' is None.")
					# Add more checks if needed, e.g., cpd.values summing to 1 across states
				except Exception as e_cpd:
					print(f"  Could not get or verify CPD for node '{node_name}': {e_cpd}")
			return None # Treat inconsistency as a failure for now

	except Exception as e:
		print(f"ERROR: Failed to fit Bayesian Network: {e}")
		# Print details about node states if fitting fails, as this is a common issue
		for node_name in nodes_for_fitting:
			if node_name in data_for_fitting:
				unique_states = data_for_fitting[node_name].unique()
				print(f"  Node '{node_name}' has states in data: {np.sort(unique_states)}")
		return None

	print(f"Table for {DEPENDENT_VAR_NAME}:")
	for cpd in bn_model.cpds:
		if cpd.variable == DEPENDENT_VAR_NAME:
			print(cpd)
			break
		
	return bn_model

def save_network_plot(model, filepath):
	"""
	Saves a plot of the Bayesian Network.
	Model is expected to be a fitted pgmpy.models.BayesianNetwork instance.
	"""
	print(f"\nSaving network plot to '{filepath}'...")
	if not model or not isinstance(model, DiscreteBayesianNetwork) or not model.nodes():
		print("Cannot plot: model is not a valid, non-empty BayesianNetwork instance.")
		return

	plt.figure(figsize=(14, 12)) # Increased size for better readability
	
	# Use networkx for layout and drawing
	graph = nx.DiGraph(model.edges())
	
	# Ensure all nodes from the model are in the graph, even isolated ones
	for node in model.nodes():
		if node not in graph:
			graph.add_node(node)

	if not graph.nodes():
		print("Graph has no nodes to plot (this shouldn't happen if model has nodes).")
		plt.close()
		return

	pos = nx.circular_layout(graph)
	#pos = nx.spring_layout(graph, pos=pos, k=2)

	nx.draw(graph, pos, with_labels=True, node_size=3500, node_color="skyblue", 
			font_size=9, font_weight="bold", arrowsize=20, width=1.5,
			edge_color="gray", style="solid") # Added edge_color and style
	plt.title(f"Learned Bayesian Network Structure ({DEPENDENT_VAR_NAME} Analysis)", fontsize=16)
	plt.tight_layout() # Adjust layout to prevent labels from overlapping
	try:
		plt.savefig(filepath, dpi=150) # Increased DPI for better quality
		print(f"Network plot saved to {filepath}")
	except Exception as e:
		print(f"Error saving plot: {e}")
	plt.close()


def plot_variable_effects(model, data, independent_vars, dependent_var, num_quantiles_config, output_dir):
	"""
	For each independent var, plots a heatmap of how its change affects the yield ratio's
	quantile probabilities, fixing other variables to their median quantile.
	'data' here is the discretized DataFrame.
	'num_quantiles_config' is the original target number of quantiles.
	"""
	print("\nPlotting variable effects on yield ratio (as heatmaps)...")
	if not model or not isinstance(model, DiscreteBayesianNetwork) or not model.nodes() or not model.get_cpds():
		print("Model not available, not fitted, or not a DiscreteBayesianNetwork. Skipping effect plotting.")
		return

	try:
		infer = VariableElimination(model)
	except Exception as e:
		print(f"Failed to initialize inference engine: {e}. Skipping effect plotting.")
		return

	base_quantile_labels = [f'q{i}' for i in range(num_quantiles_config)]
	target_median_quantile_label = base_quantile_labels[num_quantiles_config // 2]

	# Get the actual states of the dependent variable from the model
	# These will be the y-axis labels for the heatmap.
	try:
		dep_var_model_states = sorted(model.get_cpds(dependent_var).state_names[dependent_var])
	except Exception as e:
		print(f"Error getting states for dependent variable '{dependent_var}': {e}. Skipping plots.")
		return
	num_dep_var_states = len(dep_var_model_states)


	for var_to_change in independent_vars:
		if var_to_change not in model.nodes():
			print(f"  Variable '{var_to_change}' not in the learned model, skipping its heatmap.")
			continue

		print(f"  Generating heatmap for effect of '{var_to_change}' on '{dependent_var}'...")
		
		var_actual_states = sorted(data[var_to_change].unique().tolist())
		if not var_actual_states:
			print(f"	Warning: No unique states for '{var_to_change}'. Skipping.")
			continue
		
		num_var_to_change_states = len(var_actual_states)
		heatmap_data = np.zeros((num_var_to_change_states, num_dep_var_states))

		for i, current_var_state in enumerate(var_actual_states):
			evidence = {var_to_change: current_var_state}
			for other_var in independent_vars:
				if other_var not in INFERENCE_CONDITION:
					continue
				if other_var == var_to_change or other_var not in model.nodes():
					continue
				other_var_actual_states = sorted(data[other_var].unique().tolist())
				if not other_var_actual_states: continue
				
				chosen_median_state = target_median_quantile_label
				if target_median_quantile_label not in other_var_actual_states:
					chosen_median_state = other_var_actual_states[len(other_var_actual_states) // 2]
				evidence[other_var] = chosen_median_state
			
			if dependent_var not in model.nodes():
				heatmap_data[i, :] = np.nan # Fill row with NaNs
				continue

			try:
				evidence_for_query = {k: v for k, v in evidence.items() if k in model.nodes()}
				if not evidence_for_query or var_to_change not in evidence_for_query:
					heatmap_data[i, :] = np.nan
					continue

				query_result_phi = infer.query(variables=[dependent_var], evidence=evidence_for_query, show_progress=False)
				#print(f"	Query result for {evidence}:\n{query_result_phi}")
				
				# Map probabilities to the correct column in heatmap_data
				# query_result_phi.values are ordered according to model.get_cpds(dependent_var).state_names[dependent_var]
				# which we've sorted into dep_var_model_states.
				# We need to ensure the order of probabilities matches the order of dep_var_model_states.
				current_phi_states = model.get_cpds(dependent_var).state_names[dependent_var]
				prob_dict = {state: prob for state, prob in zip(current_phi_states, query_result_phi.values)}
				
				for j, dep_state_label in enumerate(dep_var_model_states):
					heatmap_data[i, j] = prob_dict.get(dep_state_label, np.nan) # Get prob by sorted state label

			except Exception as e_infer:
				print(f"	Error during inference for {var_to_change}={current_var_state}: {e_infer}")
				heatmap_data[i, :] = np.nan
		
		# Plotting the heatmap
		plt.figure(figsize=(10, 8))
		# We want var_to_change states on X-axis, dependent_var states on Y-axis.
		# heatmap_data has rows = var_to_change_states, cols = dependent_var_states.
		# So, imshow(heatmap_data.T) will put dependent_var_states on Y and var_to_change_states on X.
		plt.imshow(heatmap_data.T, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
		
		plt.colorbar(label=f"P({dependent_var} state | {var_to_change} state)")
		
		plt.title(f"Probability of '{dependent_var}' States vs. '{var_to_change}' States\n({INFERENCE_CONDITION} at median)", fontsize=14)
		
		plt.xlabel(f"State of '{var_to_change}'", fontsize=12)
		plt.xticks(ticks=np.arange(num_var_to_change_states), labels=var_actual_states, rotation=45, ha="right")
		
		plt.ylabel(f"State of '{dependent_var}'", fontsize=12)
		plt.yticks(ticks=np.arange(num_dep_var_states), labels=dep_var_model_states)
		
		plt.tight_layout()
		
		plot_filename = os.path.join(output_dir, f"heatmap-{var_to_change.replace('_','-')}-vs-{dependent_var.replace('_','-')}.png")
		try:
			plt.savefig(plot_filename, dpi=120)
			print(f"	Heatmap saved to {plot_filename}")
		except Exception as e_save:
			print(f"	Error saving heatmap {plot_filename}: {e_save}")
		plt.close()


def main_procedure():
	"""
	Main function to run the causal inference pipeline
	"""
	# Pass N_QUANTILES (the configuration for target quantiles) to preprocessing
	discretized_df = load_and_preprocess_data(DATASET_PATH, N_QUANTILES)
	if discretized_df is None or discretized_df.empty:
		print("Halting execution due to data loading/preprocessing errors.")
		return
	if TRY_ALL_CROPS and len(discretized_df) < 100:
		print("Halting execution due to small dataset size.")
		return
	
	test_data = None
	if TRY_ALL_CROPS: 
		discretized_df, test_data = train_test_split(discretized_df, test_size=0.2, random_state=42)

	# Causal Discovery
	causal_model_structure_dag = discover_causal_structure_pc(discretized_df, INDEPENDENT_VARS, DEPENDENT_VAR_NAME)
	
	if causal_model_structure_dag is None:
		print("Causal discovery did not yield a valid DAG structure. Cannot proceed with training.")
		return

	# Train Bayesian Network
	bn_model = train_bayesian_network(discretized_df, causal_model_structure_dag, INDEPENDENT_VARS, DEPENDENT_VAR_NAME)

	if bn_model is None:
		print("Halting execution due to Bayesian Network training errors.")
		return

	if TRY_ALL_CROPS: 
		print("Evaluating the accuracy score")
		with open(CROP_ACC_PATH, "a") as file:
			infer = VariableElimination(bn_model)

			y_true, y_pred = [], []
			for _, row in test_data.iterrows(): 
				try:
					evidence = row.drop(DEPENDENT_VAR_NAME).to_dict()
					true_value = int(row[DEPENDENT_VAR_NAME][1]) #conversion str quantil "qN" -> int N
					pred_dist = infer.query([DEPENDENT_VAR_NAME], evidence=evidence, show_progress=False)
					pred = pred_dist.values.argmax()
				except KeyError as e: # may happen for small datasets when a value appears in the test set and never in the train set
					print(f"KeyError: {e}")
					continue
				y_true.append(true_value)
				y_pred.append(pred)
			
			file.write(SEL_CROP + ";")
			file.write(str(len(discretized_df)) + ";")
			file.write(str(len(y_true)) + ";")
			file.write(str(accuracy_score(y_true, y_pred)) + "\n")

	else:
		# Save network plot
		save_network_plot(bn_model, NETWORK_PLOT_PATH)

		# Plot variable effects
		# Pass N_QUANTILES (config) for y-axis scaling and median calculation reference
		plot_variable_effects(bn_model, discretized_df,
						INDEPENDENT_VARS, DEPENDENT_VAR_NAME, N_QUANTILES, OUTPUT_DIR)

if __name__ == '__main__':
	ensure_output_dir()
	if TRY_ALL_CROPS:
		crops_avail = get_all_crops_available(DATASET_PATH)
		if len(crops_avail) == 0: exit()

		with open(CROP_ACC_PATH, "w") as file:
			file.write("crop;n_train;n_test;test_accuracy\n")

		for i, crop_to_test in enumerate(crops_avail): 
			print("\n\n\n\n\nCrop N ", i +1, ":\n\n\n\n\n")
			SEL_CROP = crop_to_test
			main_procedure()

		print("All crops tested.")

	else:   main_procedure()
