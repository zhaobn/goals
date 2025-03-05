import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def softmax_to_log_prob(values):
    """Convert values to log probabilities using softmax."""
    # Subtract max for numerical stability
    values = values - np.max(values)
    # Convert to probabilities
    exp_values = np.exp(values)
    probs = exp_values / np.sum(exp_values)
    # Convert to log probabilities
    return np.log(probs)

def load_probabilities():
    """Load probability data from all three policies."""
    random_probs = pd.read_csv('random_state_probabilities.csv')
    pcfg_probs = pd.read_csv('pcfg_probabilities.csv')
    
    # Load optimal probs and filter out summary rows
    optimal_probs = pd.read_csv('value-functions/goal_value_function_mean.csv')
    optimal_probs = optimal_probs[~optimal_probs['shape1_sides'].str.contains('SUMMARY', na=False)]
    
    # Convert optimal values to log probabilities using softmax
    values = optimal_probs['value'].values
    log_probs = softmax_to_log_prob(values)
    
    # Add log probabilities to dataframe
    optimal_probs['log_probability'] = log_probs
    
    return random_probs, pcfg_probs, optimal_probs

def state_to_string(row):
    """Convert a state row to a readable string."""
    return (f"{row['shape1_sides']}/{row['shape1_shade']}/{row['shape1_texture']}, "
            f"{row['shape2_sides']}/{row['shape2_shade']}/{row['shape2_texture']}, "
            f"{row['shape3_sides']}/{row['shape3_shade']}/{row['shape3_texture']}")

def compare_probabilities(random_probs, pcfg_probs, optimal_probs):
    """Compare probabilities across policies."""
    # Merge dataframes
    merged_df = random_probs.merge(
        pcfg_probs,
        on=['shape1_sides', 'shape1_shade', 'shape1_texture',
            'shape2_sides', 'shape2_shade', 'shape2_texture',
            'shape3_sides', 'shape3_shade', 'shape3_texture'],
        suffixes=('_random', '_pcfg')
    ).merge(
        optimal_probs[['shape1_sides', 'shape1_shade', 'shape1_texture',
                      'shape2_sides', 'shape2_shade', 'shape2_texture',
                      'shape3_sides', 'shape3_shade', 'shape3_texture',
                      'log_probability']],
        on=['shape1_sides', 'shape1_shade', 'shape1_texture',
            'shape2_sides', 'shape2_shade', 'shape2_texture',
            'shape3_sides', 'shape3_shade', 'shape3_texture']
    )
    
    # Rename optimal column
    merged_df = merged_df.rename(columns={'log_probability': 'log_probability_optimal'})
    
    # Find states where random has higher probability
    random_wins_pcfg = merged_df[merged_df['log_probability_random'] > merged_df['log_probability_pcfg']]
    random_wins_optimal = merged_df[merged_df['log_probability_random'] > merged_df['log_probability_optimal']]
    random_wins_both = merged_df[
        (merged_df['log_probability_random'] > merged_df['log_probability_pcfg']) &
        (merged_df['log_probability_random'] > merged_df['log_probability_optimal'])
    ]
    
    # Print results
    print(f"\nTotal states analyzed: {len(merged_df)}")
    print(f"States where random policy has higher probability than PCFG: {len(random_wins_pcfg)}")
    print(f"States where random policy has higher probability than optimal: {len(random_wins_optimal)}")
    print(f"States where random policy has higher probability than both: {len(random_wins_both)}")
    
    if len(random_wins_both) > 0:
        print("\nRandomly sampled states where random policy wins against both:")
        # Sample up to 5 random states (or fewer if there aren't 5)
        n_samples = min(5, len(random_wins_both))
        sampled_states = random_wins_both.sample(n=n_samples, random_state=42)  # Set random_state for reproducibility
        
        for _, row in sampled_states.iterrows():
            print(f"\nState: {state_to_string(row)}")
            print(f"Random log prob: {row['log_probability_random']:.3f}")
            print(f"PCFG log prob: {row['log_probability_pcfg']:.3f}")
            print(f"Optimal log prob: {row['log_probability_optimal']:.3f}")
            print(f"Difference from PCFG: {(row['log_probability_random'] - row['log_probability_pcfg']):.3f}")
            print(f"Difference from Optimal: {(row['log_probability_random'] - row['log_probability_optimal']):.3f}")
    
    return merged_df

def plot_probability_comparison(merged_df):
    """Create visualization of probability comparisons."""
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create scatter plots
    plt.scatter(merged_df['log_probability_pcfg'], 
               merged_df['log_probability_random'],
               alpha=0.5, label='PCFG vs Random', color='blue')
    plt.scatter(merged_df['log_probability_optimal'],
               merged_df['log_probability_random'],
               alpha=0.5, label='Optimal vs Random', color='green')
    
    # Add diagonal line
    min_val = min(merged_df['log_probability_random'].min(),
                 merged_df['log_probability_pcfg'].min(),
                 merged_df['log_probability_optimal'].min())
    max_val = max(merged_df['log_probability_random'].max(),
                 merged_df['log_probability_pcfg'].max(),
                 merged_df['log_probability_optimal'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('PCFG/Optimal Log Probability')
    plt.ylabel('Random Log Probability')
    plt.title('Comparison of Log Probabilities Across Policies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('figures/probability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load data
    random_probs, pcfg_probs, optimal_probs = load_probabilities()
    
    # Compare probabilities
    merged_df = compare_probabilities(random_probs, pcfg_probs, optimal_probs)
    
    # Create visualization
    plot_probability_comparison(merged_df)
    
    # Calculate and print summary statistics
    print("\nSummary Statistics:")
    print("\nRandom Policy:")
    print(merged_df['log_probability_random'].describe())
    print("\nPCFG Policy:")
    print(merged_df['log_probability_pcfg'].describe())
    print("\nOptimal Policy:")
    print(merged_df['log_probability_optimal'].describe())

if __name__ == "__main__":
    main()
