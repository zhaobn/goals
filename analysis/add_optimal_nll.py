import pandas as pd
import numpy as np
from scipy.optimize import minimize

def softmax(values, beta):
    """Compute softmax probabilities for values with inverse temperature beta."""
    exp_values = np.exp(beta * values)
    return exp_values / np.sum(exp_values)

def negative_log_likelihood(beta, values, chosen_indices):
    """Compute negative log likelihood of choices under softmax policy."""
    total_nll = 0
    for chosen_idx in chosen_indices:
        probs = softmax(values, beta)
        # Add small epsilon to prevent log(0)
        total_nll -= np.log(probs[chosen_idx] + 1e-10)
    return total_nll

def find_goal_index(row, value_function_df):
    """Find the index of a goal configuration in the value function dataframe."""
    # Create query for each shape
    query = True
    for i in range(3):  # For each shape position
        for prop in ['type', 'shade', 'texture']:
            # Map column names between dataframes
            selected_col = f'shape_{i}_{prop}'
            value_col = f'shape{i+1}_{"sides" if prop == "type" else prop}'
            
            value = row[selected_col]
            
            # Handle shape type mapping
            if prop == 'type':
                if value == 'star':  # Map star to triangle
                    value = 'triangle'
                elif value == 'circle':
                    value = 'circle'
                elif value == 'square':
                    value = 'square'
            
            query &= (value_function_df[value_col] == value)
            
            # Debug print for this specific configuration
            if row['participant_id'] == 'lhyr2v0802r5eml5u0y2k1g8' and row['trial_number'] == 5:
                print(f"\nLooking for {value_col} = {value}")
                matching_this_prop = value_function_df[value_function_df[value_col] == value]
                print(f"Found {len(matching_this_prop)} rows matching this property")
                if len(matching_this_prop) == 0:
                    print("Sample of value function column:")
                    print(value_function_df[value_col].unique()[:5])
    
    matching_rows = value_function_df[query]
    
    if len(matching_rows) == 0:
        print(f"\nNo matching configuration found for:")
        print(f"Participant: {row['participant_id']}, Trial: {row['trial_number']}")
        for i in range(3):
            print(f"Shape {i}: {row[f'shape_{i}_type']}, {row[f'shape_{i}_shade']}, {row[f'shape_{i}_texture']}")
        
        # Print value function info
        print("\nValue function columns:", value_function_df.columns)
        print("\nUnique values in relevant columns:")
        for i in range(3):
            for prop in ['sides', 'shade', 'texture']:
                col = f'shape{i+1}_{prop}'
                print(f"{col}:", value_function_df[col].unique())
        
        raise ValueError("No matching configuration found in value function")
    
    return matching_rows.index[0]

def process_participant_data(participant_goals, value_function_df):
    """Process data for a single participant."""
    # Get values for each chosen goal
    chosen_indices = []
    for _, row in participant_goals.iterrows():
        idx = find_goal_index(row, value_function_df)
        chosen_indices.append(idx)
    
    # Get all possible values
    values = value_function_df['value'].values
    
    # Find optimal beta
    result = minimize(
        lambda beta: negative_log_likelihood(beta, values, chosen_indices),
        x0=1.0,  # Initial guess
        bounds=[(0.0001, 100.0)]  # Constrain beta to be positive
    )
    
    optimal_beta = result.x[0]
    optimal_total_nll = result.fun  # Renamed from min_nll
    
    # Calculate NLL for each choice with optimal beta
    optimal_choice_nlls = []  # Renamed from choice_nlls
    for idx in chosen_indices:
        probs = softmax(values, optimal_beta)
        optimal_choice_nlls.append(-np.log(probs[idx] + 1e-10))
    
    return optimal_beta, optimal_total_nll, optimal_choice_nlls

def main():
    # Load data
    selected_goals = pd.read_csv('../data-processed/selected_goals.csv')
    value_function = pd.read_csv('../../simulations/value-functions/goal_value_function_mean.csv')
    
    # Remove summary rows from value function
    value_function = value_function[value_function['shape1_sides'] != 'SUMMARY']
    
    # Process each participant
    results = []
    for participant_id in selected_goals['participant_id'].unique():
        participant_goals = selected_goals[selected_goals['participant_id'] == participant_id]
        
        optimal_beta, optimal_total_nll, optimal_choice_nlls = process_participant_data(participant_goals, value_function)
        
        # Add results for each trial
        for (_, row), optimal_choice_nll in zip(participant_goals.iterrows(), optimal_choice_nlls):
            results.append({
                'participant_id': participant_id,
                'trial_number': row['trial_number'],
                'optimal_beta': optimal_beta,
                'optimal_total_nll': optimal_total_nll,  # Renamed
                'optimal_choice_nll': optimal_choice_nll  # Renamed
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Merge with original selected_goals
    final_df = pd.merge(
        selected_goals,
        results_df,
        on=['participant_id', 'trial_number']
    )
    
    # Save results
    final_df.to_csv('../data-processed/selected_goals_with_nll.csv', index=False)
    
    # Print summary statistics
    print("\nSummary of optimal betas:")
    print(results_df.groupby('participant_id')['optimal_beta'].first().describe())
    
    print("\nSummary of optimal total NLL per participant:")
    print(results_df.groupby('participant_id')['optimal_total_nll'].first().describe())  # Updated column name

if __name__ == "__main__":
    main()
