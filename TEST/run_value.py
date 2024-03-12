#%%
import pandas as pd
import numpy as np

def run_expectancy_table(df):
    # Filter for final pitches in at-bats and innings less than 9
    filtered_df = df[(df['final_pitch_at_bat'] == 1) & (df['inning'] < 9)]
    # Group by base_out_state, calculate mean runs_to_end_inning, and sort
    re_table = (filtered_df.groupby('base_out_state', as_index=False)
                .agg(avg_re=('runs_to_end_inning', 'mean'))
                .sort_values(by='avg_re', ascending=False))
        
df = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2022.csv")
#df = df[df['pitch_type'] == 'FF']

df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
df['final_pitch_game'] = np.where(df.groupby('game_pk')['pitch_number'].transform('max') == df['pitch_number'], 1, 0)

df = df.sort_values(by=['game_pk', 'inning_topbot', 'at_bat_number', 'pitch_number'])
df['runs_scored_on_pitch'] = df['des'].str.count("scores")
df['runs_scored_on_pitch'] = np.where(df['events'] == "home_run", df['runs_scored_on_pitch'] + 1, df['runs_scored_on_pitch'])
df['bat_score_after'] = df['bat_score'] + df['runs_scored_on_pitch']

df['final_pitch_at_bat'] = np.where(df.groupby(['game_pk', 'at_bat_number'])['pitch_number'].transform('max') == df['pitch_number'], 1, 0)
df['final_pitch_inning'] = np.where((df['final_pitch_at_bat'] == 1) & (df['inning_topbot'] != df['inning_topbot'].shift(-1)), 1, 0)
df['final_pitch_inning'] = np.where(df['final_pitch_inning'].isna(), 1, df['final_pitch_inning'])



df['bat_score_start_inning'] = df.groupby(['game_pk', 'inning', 'inning_topbot'])['bat_score'].transform('min')
df['bat_score_end_inning'] = df.groupby(['game_pk', 'inning', 'inning_topbot'])['bat_score'].transform('max')
df['cum_runs_in_inning'] = df.groupby(['game_pk', 'inning', 'inning_topbot'])['runs_scored_on_pitch'].cumsum()
df['runs_to_end_inning'] = df['bat_score_end_inning'] - df['bat_score']

df['base_out_state'] = df.apply(lambda row: f"{row['outs_when_up']} outs, " +
    ("1b" if pd.notna(row['on_1b']) else "__") + ", " +
    ("2b" if pd.notna(row['on_2b']) else "__") + ", " +
    ("3b" if pd.notna(row['on_3b']) else "__"), axis=1)

# Split the 'base_out_state' into separate components
base_states = df['base_out_state'].str.split(', ', expand=True)

df['outs'] = base_states[0]
df['on_1b'] = base_states[1]
df['on_2b'] = base_states[2]
df['on_3b'] = base_states[3]
df['base_state'] = df['on_1b'] + ' , ' + df['on_2b'] + ' , ' + df['on_3b']

filtered_df = df[(df['final_pitch_at_bat'] == 1) & (df['inning'] < 9)]

re_table = (filtered_df.groupby('base_out_state', as_index=False)
    .agg(avg_re=('runs_to_end_inning', 'mean'))
    .sort_values(by='avg_re', ascending=False))

re_table_cttable = (filtered_df.groupby(['outs', 'base_state'], as_index=False)
    .agg(avg_re=('runs_to_end_inning', 'mean'))
    .sort_values(by='avg_re', ascending=False))

# Estibalsh the pivot table of re_table
re_table_cttable['outs'] = re_table_cttable['outs'].str.strip()
pivot_re_table = re_table_cttable.pivot(index='base_state', \
columns='outs', values='avg_re').fillna(0)

print(pivot_re_table)
#%%
##################################################
# 將df增加資料列
df = pd.merge(df, re_table, how='left', on='base_out_state')
df = df[df['final_pitch_at_bat'] == 1]
df['next_base_out_state'] = df.groupby(['game_pk', 'inning', 'inning_topbot'])['base_out_state'].shift(-1)
df = pd.merge(df, re_table, how='left', left_on='next_base_out_state', right_on='base_out_state', suffixes=('', '_next'))

df['next_avg_re'] = df['avg_re_next'].fillna(0)
df['change_re'] = df['next_avg_re'] - df['avg_re']
df['re24'] = df['change_re'] + df['runs_scored_on_pitch']

df.sort_values(by=['game_pk', 'inning', 'inning_topbot'], inplace=True)
########################################################

#%%