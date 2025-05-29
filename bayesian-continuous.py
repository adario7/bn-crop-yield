import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.estimators import HillClimbSearch, ExpertKnowledge
from pgmpy.inference import VariableElimination

# Configuration
MAX_INDEGREE = None
ONE_CROP = "PEAR"
RATIO = 'productive_surface' # 'tot_surface', 'productive_surface' or None

DATASET_PATH = 'build/dataset.csv.gz'
OUTPUT_DIR = 'build/'
NETWORK_PLOT_PATH = os.path.join(OUTPUT_DIR, 'bayesian-network.png')

# Define independent and dependent variables
VAR_TEMP = [ 'mean_temperature', 'precipitation', 'heat_days', 'heavy_rain_days', 'dry_days' ]
VAR_SOIL = [ 'soil_anhy_fosfor', 'soil_mcrnt_ox_potas', 'soil_nitrogen', 'soil_org_comp' ]
INDEPENDENT_VARS = VAR_TEMP + VAR_SOIL
DEPENDENT_VAR_NAME = 'yield_ratio'

FORBIDDEN_EDGES = [ (a, b) for a in VAR_TEMP for b in VAR_SOIL ] \
	+ [ (b, a) for a in VAR_TEMP for b in VAR_SOIL ] \
	+ [ (a, b) for a in VAR_SOIL for b in VAR_SOIL if a != b ] \
	+ [ (DEPENDENT_VAR_NAME, a) for a in INDEPENDENT_VARS ]

INFERENCE_CONDITION = VAR_TEMP

def ensure_output_dir():
	"""Ensures the output directory exists."""
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	print(f"Output directory '{OUTPUT_DIR}' ensured.")

def load_and_preprocess_data(filepath):
	"""
	Loads data, selects the most frequent crop, calculates yield_ratio.
	No discretization for continuous BN.
	"""
	print(f"Loading data from '{filepath}'...")
	try:
		with gzip.open(filepath, 'rt') as f:
			df = pd.read_csv(f)
	except FileNotFoundError:
		print(f"ERROR: Dataset not found at {filepath}. Please ensure it exists.")
		return None, None, None

	print(f"Initial dataset shape: {df.shape}")

	# Find the crop with the most rows
	print(df['crop'].value_counts().head(10))
	if isinstance(ONE_CROP, str):
		most_frequent_crop = ONE_CROP
	else:
		most_frequent_crop = df['crop'].mode()[0]
	print(f"Most frequent crop: '{most_frequent_crop}'")
	
	if ONE_CROP is True or isinstance(ONE_CROP, str):
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
	df_crop.dropna(subset=all_vars_for_analysis, inplace=True)
	print(f"Shape after dropping NaNs in relevant columns ({len(all_vars_for_analysis)} cols): {df_crop.shape}")

	if df_crop.empty:
		print("ERROR: No data remaining after preprocessing (NaN drop). Cannot proceed.")
		return None, None, None

	print("Data sample:")
	print(df_crop[all_vars_for_analysis].describe())

	return df_crop[all_vars_for_analysis], INDEPENDENT_VARS, DEPENDENT_VAR_NAME

def discover_causal_structure_pc(data, independent_vars, dependent_var):
	"""
	Discovers causal structure using Hill Climb Search for continuous data.
	"""
	print("\nDiscovering causal structure using Hill Climb Search...")
	all_vars = independent_vars + [dependent_var]
	data_for_hc = data[all_vars].copy()

	try:
		# scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g, ll-cg, aic-cg, bic-cg
		hc_estimator = HillClimbSearch(data=data_for_hc)
		model_dag = hc_estimator.estimate(
			tabu_length=1000, 
			max_indegree=MAX_INDEGREE, 
			expert_knowledge=ExpertKnowledge(forbidden_edges=FORBIDDEN_EDGES), 
			scoring_method="bic-g"
		)
		print("HillClimbSearch with BIC scoring completed.")
	except Exception as e_hc:
		print(f"HillClimbSearch failed: {e_hc}")
		print("ERROR: Could not determine causal structure. Aborting.")
		return None

	if not model_dag.edges():
		print("Warning: Structure learning algorithm did not find any causal edges.")
		model_dag.add_nodes_from(all_vars)

	print("Discovered causal edges:")
	for edge in model_dag.edges():
		print(f"  {edge[0]} -> {edge[1]}")
	
	return model_dag

def train_bayesian_network(data, structure_dag, independent_vars, dependent_var):
	"""
	Trains a Linear Gaussian Bayesian Network.
	"""
	print("\nTraining Linear Gaussian Bayesian Network...")
	
	# Create Linear Gaussian BN
	bn_model = LinearGaussianBayesianNetwork() 
	bn_model.add_nodes_from(structure_dag.nodes())
	bn_model.add_edges_from(structure_dag.edges())

	nodes_for_fitting = list(bn_model.nodes())
	data_for_fitting = data[nodes_for_fitting]

	try:
		bn_model.fit(data_for_fitting)
		print("Linear Gaussian Bayesian Network training complete.")
		
		# Print learned parameters for dependent variable
		if dependent_var in bn_model.nodes():
			cpd = bn_model.get_cpds(dependent_var)
			print(f"\nLinear model for {dependent_var}:")
			print(f"  Mean: {cpd}")
			#print(f"  Mean: {cpd.mean}")
			#print(f"  Variance: {cpd.variance}")
			#if hasattr(cpd, 'beta'):
				#print(f"  Beta coefficients: {cpd.beta}")

	except Exception as e:
		print(f"ERROR: Failed to fit Linear Gaussian Bayesian Network: {e}")
		return None
		
	return bn_model

def save_network_plot(model, filepath):
	"""Saves a plot of the Bayesian Network."""
	print(f"\nSaving network plot to '{filepath}'...")
	if not model or not model.nodes():
		print("Cannot plot: model is not valid or empty.")
		return

	plt.figure(figsize=(14, 12))
	
	graph = nx.DiGraph(model.edges())
	for node in model.nodes():
		if node not in graph:
			graph.add_node(node)

	if not graph.nodes():
		print("Graph has no nodes to plot.")
		plt.close()
		return

	pos = nx.circular_layout(graph)
	nx.draw(graph, pos, with_labels=True, node_size=3500, node_color="lightcoral", 
			font_size=9, font_weight="bold", arrowsize=20, width=1.5,
			edge_color="gray", style="solid")
	plt.title(f"Learned Linear Gaussian Bayesian Network ({DEPENDENT_VAR_NAME} Analysis)", fontsize=16)
	plt.tight_layout()
	
	try:
		plt.savefig(filepath, dpi=150)
		print(f"Network plot saved to {filepath}")
	except Exception as e:
		print(f"Error saving plot: {e}")
	plt.close()


def predict_with_linear_gaussian(model, target_var, evidence_df):
    """
    Prediction for LinearGaussianBayesianNetwork using the correct predict method.
    
    Parameters:
    - model: The trained Linear Gaussian Bayesian Network
    - target_var: The variable to predict
    - evidence_df: DataFrame with evidence variables (missing the target variable)
    
    Returns:
    - predictions: numpy array of predicted means
    - variances: numpy array of predicted variances
    """
    if target_var not in model.nodes():
        return None, None
    
    try:
        # Use the model's predict method
        variables, mu, cov = model.predict(evidence_df)
        
        # Check if our target variable is in the predicted variables
        if target_var in variables:
            target_idx = variables.index(target_var)
            predictions = mu[:, target_idx]
            # Extract variance for the target variable (diagonal of covariance matrix)
            if cov.ndim == 3:  # Multiple predictions
                variances = cov[:, target_idx, target_idx]
            else:  # Single covariance matrix
                variances = np.full(len(predictions), cov[target_idx, target_idx])
        else:
            print(f"Target variable {target_var} not found in prediction results")
            return None, None
            
        return predictions, variances
        
    except Exception as e:
        print(f"Error in prediction for {target_var}: {e}")
        return None, None


def plot_variable_effects(model, data, independent_vars, dependent_var, output_dir):
    """
    For continuous BN, plot regression-style effects of each variable.
    Uses the correct predict method for Linear Gaussian Bayesian Networks.
    """
    print("\nPlotting variable effects for continuous Bayesian Network...")
    if not model or not model.nodes():
        print("Model not available. Skipping effect plotting.")
        return

    # Get all variables that are actually in the model
    model_vars = list(model.nodes())
    available_independent_vars = [var for var in independent_vars if var in model_vars and var in INFERENCE_CONDITION]
    
    if dependent_var not in model_vars:
        print(f"Dependent variable '{dependent_var}' not in the learned model. Cannot plot effects.")
        return
    
    print(f"Available independent variables in model: {available_independent_vars}")

    for var_to_change in available_independent_vars:
        print(f"  Generating effect plot for '{var_to_change}' on '{dependent_var}'...")
        
        # Create range of values for the variable
        var_min, var_max = data[var_to_change].min(), data[var_to_change].max()
        var_range = np.linspace(var_min, var_max, 50)
        
        # Create base evidence DataFrame with median values for other variables
        evidence_data = []
        for val in var_range:
            evidence_row = {}
            # Set the variable of interest to the current value
            evidence_row[var_to_change] = val
            
            # Set other variables to their median values
            for other_var in available_independent_vars:
                if other_var != var_to_change:
                    evidence_row[other_var] = data[other_var].median()
            
            evidence_data.append(evidence_row)
        
        # Create DataFrame for prediction (excluding the dependent variable)
        evidence_df = pd.DataFrame(evidence_data)
        
        # Make predictions
        predictions, variances = predict_with_linear_gaussian(model, dependent_var, evidence_df)
        
        if predictions is None:
            print(f"    Could not generate predictions for {var_to_change}. Skipping plot.")
            continue
        
        # Create the effect plot
        plt.figure(figsize=(10, 6))
        
        # Handle potential NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(variances))
        if not np.any(valid_mask):
            print(f"    All predictions are NaN for {var_to_change}. Skipping plot.")
            plt.close()
            continue
        
        valid_range = var_range[valid_mask]
        valid_predictions = predictions[valid_mask]
        valid_variances = variances[valid_mask]
        
        # Calculate standard deviations
        std_devs = np.sqrt(np.abs(valid_variances))
        
        # Plot the main prediction line
        plt.plot(valid_range, valid_predictions, 'b-', linewidth=2, label='Predicted mean')
        
        # Plot confidence intervals
        plt.fill_between(valid_range, 
                        valid_predictions - std_devs, 
                        valid_predictions + std_devs, 
                        alpha=0.3, color='blue', label='±1 std dev')
        
        # Add scatter points for actual data
        plt.scatter(data[var_to_change], data[dependent_var], 
                   alpha=0.3, color='red', s=10, label='Actual data')
        
        # Formatting
        plt.xlabel(f'{var_to_change.replace("_", " ").title()}')
        plt.ylabel(f'{dependent_var.replace("_", " ").title()}')
        plt.title(f'Effect of {var_to_change.replace("_", " ").title()} on {dependent_var.replace("_", " ").title()}\n'
                 f'(Other variables fixed at median values)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_filename = os.path.join(output_dir, 
                                   f"effect-{var_to_change.replace('_','-')}-on-{dependent_var.replace('_','-')}.png")
        try:
            plt.savefig(plot_filename, dpi=120, bbox_inches='tight')
            print(f"    Effect plot saved to {plot_filename}")
        except Exception as e:
            print(f"    Error saving plot {plot_filename}: {e}")
        plt.close()


def main():
	"""Main function to run the continuous Bayesian network analysis."""
	print("Starting Continuous Bayesian Network Analysis for Agriculture")
	ensure_output_dir()

	# Load and preprocess data (no discretization)
	continuous_df, ind_vars, dep_var = load_and_preprocess_data(DATASET_PATH)

	if continuous_df is None or continuous_df.empty:
		print("Halting execution due to data loading/preprocessing errors.")
		return

	# Causal Discovery
	causal_model_structure_dag = discover_causal_structure_pc(continuous_df, ind_vars, dep_var)
	
	if causal_model_structure_dag is None:
		print("Causal discovery did not yield a valid DAG structure. Cannot proceed with training.")
		return

	# Train Bayesian Network
	bn_model = train_bayesian_network(continuous_df, causal_model_structure_dag, ind_vars, dep_var)

	if bn_model is None:
		print("Halting execution due to Bayesian Network training errors.")
		return

	# Save network plot
	save_network_plot(bn_model, NETWORK_PLOT_PATH)

	# Plot variable effects
	plot_variable_effects(bn_model, continuous_df, ind_vars, dep_var, OUTPUT_DIR)

if __name__ == '__main__':
	main()
