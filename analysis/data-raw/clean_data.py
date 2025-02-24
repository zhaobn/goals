import json
import pandas as pd
import glob
import os

def load_participant_data(file_path):
    """Load JSON data from a participant file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_goal_selection_data(trial_data):
    """Extract goal selection data from a trial."""
    if trial_data.get('trial_type') == 'goal-selection':
        selection_actions = trial_data.get('selection_actions', [])
        final_goal = trial_data.get('final_goal', [])
        
        # Create rows for selection actions
        action_rows = []
        for action in selection_actions:
            if action['action'] == 'initial_configuration':
                continue
            action_row = {
                'timestamp': action.get('timestamp'),
                'action_type': action.get('action'),
                'position': action.get('position')
            }
            if 'item' in action:
                action_row.update({
                    'shape_type': clean_property(action['item'].get('type')),
                    'shade': clean_property(action['item'].get('shade')),
                    'texture': clean_property(action['item'].get('texture'))
                })
            action_rows.append(action_row)
            
        # Create rows for final goal configuration
        goal_rows = []
        for goal in final_goal:
            if not goal.get('item'):
                continue
            goal_row = {
                'position': goal['position'],
                'shape_type': clean_property(goal['item'].get('type')),
                'shade': clean_property(goal['item'].get('shade')),
                'texture': clean_property(goal['item'].get('texture'))
            }
            goal_rows.append(goal_row)
            
        return {
            'actions': pd.DataFrame(action_rows) if action_rows else None,
            'final_goal': pd.DataFrame(goal_rows) if goal_rows else None
        }
    return None

def clean_property(value):
    """Remove prefixes from shape types and shades."""
    if not value:
        return value
    
    # Remove 'goal-' prefix from shapes
    if value.startswith('goal-'):
        value = value[5:]
    
    # Remove 'shade-' prefix from shades
    if value.startswith('shade-'):
        value = value[6:]
        
    # Map textures to match value function format
    texture_mapping = {
        'striped': 'stripes',
        'dotted': 'dots'
        # 'plain' stays as 'plain'
    }
    
    # Map shape types
    shape_mapping = {
        'star': 'triangle',
        'cloud': 'circle'
    }
    
    # Map shade names
    shade_mapping = {
        'light': 'low',
        'dark': 'high'
        # 'medium' stays as 'medium'
    }
    
    if value in texture_mapping:
        value = texture_mapping[value]
    elif value in shape_mapping:
        value = shape_mapping[value]
    elif value in shade_mapping:
        value = shade_mapping[value]
    
    return value

def extract_goal_pursuit_data(trial_data):
    """Extract goal pursuit (goal-display) data from a trial."""
    if trial_data.get('trial_type') == 'goal-display':
        pursuit_array = trial_data.get('pursuit_array', [])
        goal = trial_data.get('goal', [])
        
        # Create rows for pursuit actions
        pursuit_rows = []
        for step_num, action in enumerate(pursuit_array, 1):
            state = action.get('state', [])
            
            action_row = {
                'timestamp': action.get('timestamp'),
                'actor_position': action.get('actor_position'),
                'recipient_position': action.get('recipient_position'),
                'feature': action.get('feature'),
                'step_number': step_num
            }
            
            # Add state information for each position
            for i, state_item in enumerate(state):
                action_row.update({
                    f'state_{i}_type': clean_property(state_item.get('type')),
                    f'state_{i}_shade': clean_property(state_item.get('shade')),
                    f'state_{i}_texture': clean_property(state_item.get('texture'))
                })
            
            # Add goal information
            for i, goal_item in enumerate(goal):
                action_row.update({
                    f'goal_{i}_type': clean_property(goal_item.get('type')),
                    f'goal_{i}_shade': clean_property(goal_item.get('shade')),
                    f'goal_{i}_texture': clean_property(goal_item.get('texture'))
                })
            
            pursuit_rows.append(action_row)
            
        return {
            'pursuit': pd.DataFrame(pursuit_rows) if pursuit_rows else None,
            'abandoned': trial_data.get('abandoned', False),
            'goal_achieved': trial_data.get('goal_achieved', False),
            'steps': trial_data.get('steps', 0)
        }
    return None

def extract_debrief_data(trial_data):
    """Extract debrief data from a trial."""
    # Print trial type and response for debugging
    print(f"Checking for debrief data in trial type: {trial_data.get('trial_type')}")
    
    # Check for survey trial types including survey-html-form
    if trial_data.get('trial_type') in ['survey-likert', 'survey-text', 'survey', 'survey-html-form']:
        responses = trial_data.get('response', {})
        if isinstance(responses, dict):
            # Add trial type to responses for context
            responses['trial_type'] = trial_data.get('trial_type')
            print(f"Processing debrief response: {responses}")
            return pd.DataFrame([responses])
        elif isinstance(responses, (str, int, float)):
            # Handle single response values
            return pd.DataFrame([{
                'response': responses,
                'trial_type': trial_data.get('trial_type')
            }])
    return None

def process_all_files():
    """Process all JSON files in the data-raw directory."""
    # Get all JSON files
    json_files = glob.glob('*.json')
    print(f"Found JSON files: {json_files}")
    
    all_goal_selections = []
    all_goal_pursuits = []
    all_debrief_responses = []
    all_selected_goals = []
    
    for file_path in json_files:
        participant_id = os.path.basename(file_path).split('.')[0]
        print(f"Processing participant {participant_id}")
        data = load_participant_data(file_path)
        print(f"Found {len(data)} trials")
        
        # Keep track of goal pursuit trial number
        pursuit_trial_counter = 0
        
        for trial in data:
            print(f"Processing trial type: {trial.get('trial_type')}")
            # Add participant ID to all data
            trial['participant_id'] = participant_id
            
            # Process goal selection data
            goal_selection = extract_goal_selection_data(trial)
            if goal_selection:
                if goal_selection['actions'] is not None:
                    goal_selection['actions']['participant_id'] = participant_id
                    all_goal_selections.append(goal_selection['actions'])
                if goal_selection['final_goal'] is not None:
                    goal_selection['final_goal']['participant_id'] = participant_id
                    all_goal_selections.append(goal_selection['final_goal'])
            
            # Process goal pursuit data
            goal_pursuit = extract_goal_pursuit_data(trial)
            if goal_pursuit:
                if goal_pursuit['pursuit'] is not None:
                    pursuit_trial_counter += 1
                    
                    # Extract goal configuration for this trial
                    goal = trial.get('goal', [])
                    goal_row = {
                        'participant_id': participant_id,
                        'trial_number': pursuit_trial_counter,
                        'abandoned': goal_pursuit['abandoned'],
                        'goal_achieved': goal_pursuit['goal_achieved']
                    }
                    
                    # Add goal information
                    for i, goal_item in enumerate(goal):
                        goal_row.update({
                            f'shape_{i}_type': clean_property(goal_item.get('type')),
                            f'shape_{i}_shade': clean_property(goal_item.get('shade')),
                            f'shape_{i}_texture': clean_property(goal_item.get('texture'))
                        })
                    
                    all_selected_goals.append(goal_row)
                    
                    pursuit_df = goal_pursuit['pursuit']
                    pursuit_df['participant_id'] = participant_id
                    pursuit_df['trial_number'] = pursuit_trial_counter
                    pursuit_df['abandoned'] = goal_pursuit['abandoned']
                    pursuit_df['goal_achieved'] = goal_pursuit['goal_achieved']
                    pursuit_df['steps'] = goal_pursuit['steps']
                    
                    cols_to_drop = ['position', 'shape_type', 'shade', 'texture']
                    pursuit_df = pursuit_df.drop(columns=[col for col in cols_to_drop if col in pursuit_df.columns])
                    
                    all_goal_pursuits.append(pursuit_df)
            
            # Process debrief data
            debrief = extract_debrief_data(trial)
            if debrief is not None:
                debrief['participant_id'] = participant_id
                all_debrief_responses.append(debrief)
    
    # Create all dataframes
    goal_selections_df = pd.concat(all_goal_selections, ignore_index=True) if all_goal_selections else pd.DataFrame()
    goal_pursuits_df = pd.concat(all_goal_pursuits, ignore_index=True) if all_goal_pursuits else pd.DataFrame()
    debrief_df = pd.concat(all_debrief_responses, ignore_index=True) if all_debrief_responses else pd.DataFrame()
    selected_goals_df = pd.DataFrame(all_selected_goals) if all_selected_goals else pd.DataFrame()
    
    return goal_selections_df, goal_pursuits_df, debrief_df, selected_goals_df

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('../data-processed', exist_ok=True)
    
    # Process all files and get dataframes
    goal_selections, goal_pursuits, debrief, selected_goals = process_all_files()
    
    # Save to CSV files
    goal_selections.to_csv('../data-processed/goal_selections.csv', index=False)
    goal_pursuits.to_csv('../data-processed/goal_pursuits.csv', index=False)
    debrief.to_csv('../data-processed/debrief.csv', index=False)
    selected_goals.to_csv('../data-processed/selected_goals.csv', index=False)
