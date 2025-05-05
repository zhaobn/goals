import pandas as pd
import numpy as np
import os

def find_goal_index(row, random_probs_df):
    """Find the index of a goal configuration in the random probabilities dataframe."""
    # Create query for each shape
    query = True
    for i in range(3):  # For each shape position
        for prop in ['type', 'shade', 'texture']:
            # Map column names between dataframes
            selected_col = f'shape_{i}_{prop}'
            prob_col = f'shape{i+1}_{"sides" if prop == "type" else prop}'
            
            value = row[selected_col]
            query &= (random_probs_df[prob_col] == value)
    
    matching_rows = random_probs_df[query]
    
    if len(matching_rows) == 0:
        print(f"\nNo matching configuration found for:")
        print(f"Participant: {row['participant_id']}, Trial: {row['trial_number']}")
        for i in range(3):
            print(f"Shape {i}: {row[f'shape_{i}_type']}, {row[f'shape_{i}_shade']}, {row[f'shape_{i}_texture']}")
        raise ValueError("No matching configuration found in random probabilities")
    
    return matching_rows.index[0]

def process_participant_data(participant_goals, random_probs_df):
    """Process data for a single participant."""
    # Get random log probability for each chosen goal
    random_choice_nlls = []
    for _, row in participant_goals.iterrows():
        idx = find_goal_index(row, random_probs_df)
        # For random policy, the NLL is just the negative of the log probability
        random_choice_nlls.append(-random_probs_df.loc[idx, 'log_probability'])
    
    # Total NLL is the sum of individual choice NLLs
    random_total_nll = sum(random_choice_nlls)
    
    # Calculate BIC
    n_trials = len(random_choice_nlls)
    k_params = 0  # Random model has no parameters
    random_total_bic = 2 * random_total_nll + k_params * np.log(n_trials)
    
    # Calculate per-trial BIC (for random model, this is just 2*NLL since k=0)
    random_choice_bics = [2 * nll for nll in random_choice_nlls]
    
    return random_total_nll, random_choice_nlls, random_total_bic, random_choice_bics

def main():
    # Load data
    selected_goals = pd.read_csv('./data-processed/selected_goals.csv')
    random_probs = pd.read_csv('../simulations/random_state_probabilities.csv')
    
    # Process each participant
    results = []
    for participant_id in selected_goals['participant_id'].unique():
        participant_goals = selected_goals[selected_goals['participant_id'] == participant_id]
        
        random_total_nll, random_choice_nlls, random_total_bic, random_choice_bics = process_participant_data(
            participant_goals, random_probs
        )
        
        # Add results for each trial
        for (_, row), random_choice_nll, random_choice_bic in zip(participant_goals.iterrows(), random_choice_nlls, random_choice_bics):
            results.append({
                'participant_id': participant_id,
                'trial_number': row['trial_number'],
                'random_total_nll': random_total_nll,
                'random_choice_nll': random_choice_nll,
                'random_total_bic': random_total_bic,
                'random_choice_bic': random_choice_bic
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Check if the existing NLL file exists
    if os.path.exists('./data-processed/selected_goals_with_nll.csv'):
        # Load existing NLL results and merge
        existing_df = pd.read_csv('./data-processed/selected_goals_with_nll.csv')
        final_df = pd.merge(
            existing_df,
            results_df,
            on=['participant_id', 'trial_number']
        )
    else:
        # First time running - merge with original selected_goals
        final_df = pd.merge(
            selected_goals,
            results_df,
            on=['participant_id', 'trial_number']
        )
    
    # Save results
    final_df.to_csv('./data-processed/selected_goals_with_nll.csv', index=False)
    
    # Print summary statistics
    print("\nSummary of random total NLL per participant:")
    print(results_df.groupby('participant_id')['random_total_nll'].first().describe())
    
    print("\nSummary of random choice NLL:")
    print(results_df['random_choice_nll'].describe())
    
    print("\nSummary of random total BIC per participant:")
    print(results_df.groupby('participant_id')['random_total_bic'].first().describe())

if __name__ == "__main__":
    main()
