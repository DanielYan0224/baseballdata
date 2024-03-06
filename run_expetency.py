import pandas as pd

def compute_runs_expectancy(season):
    # Read the season data and fields mapping
    data_file = f"all{season}.csv"
    data = pd.read_csv(data_file, header=None)
    fields = pd.read_csv("https://raw.githubusercontent.com/beanumber/baseball_R/master/data/fields.csv")
    data.columns = fields['Header'].values
    
    # Calculate runs and half inning identifier
    data['RUNS'] = data['AWAY_SCORE_CT'] + data['HOME_SCORE_CT']
    data['HALF.INNING'] = data.apply(lambda row: f"{row['GAME_ID']}{row['INN_CT']}{row['BAT_HOME_ID']}", axis=1)
    
    # Calculate runs scored
    data['RUNS.SCORED'] = (data['BAT_DEST_ID'] > 3).astype(int) + \
                          (data['RUN1_DEST_ID'] > 3).astype(int) + \
                          (data['RUN2_DEST_ID'] > 3).astype(int) + \
                          (data['RUN3_DEST_ID'] > 3).astype(int)
                          
    # Aggregate runs scored and start by half inning
    runs_scored_inning = data.groupby('HALF.INNING')['RUNS.SCORED'].sum().reset_index(name='RUNS.SCORED.INNING')
    runs_scored_start = data.groupby('HALF.INNING')['RUNS'].first().reset_index(name='RUNS.SCORED.START')
    
    # Merge and calculate max runs
    data = pd.merge(data, runs_scored_inning, on='HALF.INNING', how='left')
    data = pd.merge(data, runs_scored_start, on='HALF.INNING', how='left')
    data['MAX.RUNS'] = data['RUNS.SCORED.INNING'] + data['RUNS.SCORED.START']
    
    # Calculate runs ROI
    data['RUNS.ROI'] = data['MAX.RUNS'] - data['RUNS']
    
    # Define helper function to get state
    def get_state(runner1, runner2, runner3, outs):
        runners = f"{runner1}{runner2}{runner3}"
        return f"{runners} {outs}"
    
    # Calculate current and new states
    data['STATE'] = data.apply(lambda row: get_state(row['BASE1_RUN_ID'] != "", 
                                                     row['BASE2_RUN_ID'] != "", 
                                                     row['BASE3_RUN_ID'] != "", 
                                                     row['OUTS_CT']), axis=1)
    data['NEW.STATE'] = data.apply(lambda row: get_state(row['RUN1_DEST_ID'] == 1 or row['BAT_DEST_ID'] == 1,
                                                         row['RUN1_DEST_ID'] == 2 or row['RUN2_DEST_ID'] == 2 or row['BAT_DEST_ID'] == 2,
                                                         row['RUN1_DEST_ID'] == 3 or row['RUN2_DEST_ID'] == 3 or row['RUN3_DEST_ID'] == 3 or row['BAT_DEST_ID'] == 3,
                                                         row['OUTS_CT'] + row['EVENT_OUTS_CT']), axis=1)
    
    # Filter data based on state change or runs scored
    data_filtered = data[(data['STATE'] != data['NEW.STATE']) | (data['RUNS.SCORED'] > 0)]
    
    # Calculate outs in inning using dplyr equivalent in pandas
    data_outs = data.groupby('HALF.INNING').agg({'EVENT_OUTS_CT': 'sum'}).rename(columns={'EVENT_OUTS_CT': 'Outs.Inning'}).reset_index()
    data_filtered = pd.merge(data_filtered, data_outs, on='HALF.INNING', how='left')
    
    # Filter complete innings for expected runs computation
    data_complete = data_filtered[data_filtered['Outs.Inning'] == 3]
    
    # Calculate mean runs ROI by state
    runs = data_complete.groupby('STATE')['RUNS.ROI'].mean().reset_index(name='Mean')
    runs['Outs'] = runs['STATE'].apply(lambda x: x.split(' ')[1])
    runs.sort_values(by='Outs', inplace=True)
    
    # Prepare runs potential matrix
    runs_potential = pd.concat([runs['Mean'], pd.Series([0] * 8)], ignore_index=True)
    runs_potential.index = list(runs['STATE']) + ["000 3", "001 3", "010 3", "011 3", "100 3", "101 3", "110 3", "111 3"]
    
    # Calculate runs value
    data_filtered['RUNS.STATE'] = data_filtered['STATE'].apply(lambda x: runs_potential[x])
    data_filtered['RUNS.NEW.STATE'] = data_filtered['NEW.STATE'].apply(lambda x: runs_potential.get(x, 0))
    data_filtered['RUNS.VALUE'] = data_filtered['RUNS.NEW.STATE'] - data_filtered['RUNS.STATE'] + data_filtered['RUNS.SCORED']
    
    return data_filtered

# Example usage
season_data = compute_runs_expectancy(1961)
