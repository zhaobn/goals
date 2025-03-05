import pandas as pd
import numpy as np

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
    
    return random_total_nll, random_choice_nlls

def main():
    # Load data
    selected_goals = pd.read_csv('../data-processed/selected_goals.csv')
    random_probs = pd.read_csv('../../simulations/random_state_probabilities.csv')
    
    # Process each participant
    results = []
    for participant_id in selected_goals['participant_id'].unique():
        participant_goals = selected_goals[selected_goals['participant_id'] == participant_id]
        
        random_total_nll, random_choice_nlls = process_participant_data(
            participant_goals, random_probs
        )
        
        # Add results for each trial
        for (_, row), random_choice_nll in zip(participant_goals.iterrows(), random_choice_nlls):
            results.append({
                'participant_id': participant_id,
                'trial_number': row['trial_number'],
                'random_total_nll': random_total_nll,
                'random_choice_nll': random_choice_nll
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
    print("\nSummary of random total NLL per participant:")
    print(results_df.groupby('participant_id')['random_total_nll'].first().describe())
    
    print("\nSummary of random choice NLL:")
    print(results_df['random_choice_nll'].describe())

if __name__ == "__main__":
    main()
