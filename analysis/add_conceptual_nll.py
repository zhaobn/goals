import pandas as pd
import numpy as np
from scipy.optimize import minimize

def softmax(log_probs, beta):
    """Compute softmax probabilities for log probabilities with inverse temperature beta."""
    # Convert log probabilities to values (multiply by beta)
    values = beta * np.array(log_probs)
    # Subtract max for numerical stability
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)

def negative_log_likelihood(beta, log_probs, chosen_indices):
    """Compute negative log likelihood of choices under conceptual softmax policy."""
    total_nll = 0
    for chosen_idx in chosen_indices:
        probs = softmax(log_probs, beta)
        # Add small epsilon to prevent log(0)
        total_nll -= np.log(probs[chosen_idx] + 1e-10)
    return total_nll

def find_goal_index(row, conceptual_probs_df):
    """Find the index of a goal configuration in the conceptual probabilities dataframe."""
    # Create query for each shape
    query = True
    for i in range(3):  # For each shape position
        for prop in ['type', 'shade', 'texture']:
            # Map column names between dataframes
            selected_col = f'shape_{i}_{prop}'
            prob_col = f'shape{i+1}_{"sides" if prop == "type" else prop}'
            
            value = row[selected_col]
            query &= (conceptual_probs_df[prob_col] == value)
    
    matching_rows = conceptual_probs_df[query]
    
    if len(matching_rows) == 0:
        print(f"\nNo matching configuration found for:")
        print(f"Participant: {row['participant_id']}, Trial: {row['trial_number']}")
        for i in range(3):
            print(f"Shape {i}: {row[f'shape_{i}_type']}, {row[f'shape_{i}_shade']}, {row[f'shape_{i}_texture']}")
        raise ValueError("No matching configuration found in conceptual probabilities")
    
    return matching_rows.index[0]

def process_participant_data(participant_goals, conceptual_probs_df):
    """Process data for a single participant."""
    # Get values for each chosen goal
    chosen_indices = []
    for _, row in participant_goals.iterrows():
        idx = find_goal_index(row, conceptual_probs_df)
        chosen_indices.append(idx)
    
    # Get all log probabilities
    log_probs = conceptual_probs_df['log_probability'].values
    
    # Find optimal beta
    result = minimize(
        lambda beta: negative_log_likelihood(beta, log_probs, chosen_indices),
        x0=1.0,  # Initial guess
        bounds=[(0.0001, 100.0)]  # Constrain beta to be positive
    )
    
    conceptual_beta = result.x[0]
    conceptual_total_nll = result.fun
    
    # Calculate NLL for each choice with optimal beta
    conceptual_choice_nlls = []
    for idx in chosen_indices:
        probs = softmax(log_probs, conceptual_beta)
        conceptual_choice_nlls.append(-np.log(probs[idx] + 1e-10))
    
    return conceptual_beta, conceptual_total_nll, conceptual_choice_nlls

def main():
    # Load data
    selected_goals = pd.read_csv('../data-processed/selected_goals.csv')
    conceptual_probs = pd.read_csv('../../simulations/state_probabilities.csv')
    
    # Process each participant
    results = []
    for participant_id in selected_goals['participant_id'].unique():
        participant_goals = selected_goals[selected_goals['participant_id'] == participant_id]
        
        conceptual_beta, conceptual_total_nll, conceptual_choice_nlls = process_participant_data(
            participant_goals, conceptual_probs
        )
        
        # Add results for each trial
        for (_, row), conceptual_choice_nll in zip(participant_goals.iterrows(), conceptual_choice_nlls):
            results.append({
                'participant_id': participant_id,
                'trial_number': row['trial_number'],
                'conceptual_beta': conceptual_beta,
                'conceptual_total_nll': conceptual_total_nll,
                'conceptual_choice_nll': conceptual_choice_nll
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Load existing NLL results
    existing_df = pd.read_csv('../data-processed/selected_goals_with_nll.csv')
    
    # Merge with existing results
    final_df = pd.merge(
        existing_df,
        results_df,
        on=['participant_id', 'trial_number']
    )
    
    # Save results
    final_df.to_csv('../data-processed/selected_goals_with_nll.csv', index=False)
    
    # Print summary statistics
    print("\nSummary of conceptual betas:")
    print(results_df.groupby('participant_id')['conceptual_beta'].first().describe())
    
    print("\nSummary of conceptual total NLL per participant:")
    print(results_df.groupby('participant_id')['conceptual_total_nll'].first().describe())

if __name__ == "__main__":
    main()
