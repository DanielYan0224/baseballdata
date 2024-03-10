#%%
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2019.csv")

#def run_expectancy(df, level='plate appearence'):

# 排序
df.sort_values(by = ['game_pk', 'at_bat_number', 'pitch_number'], inplace=True)

# last pitch per game
df['final_pitch_game'] = df.groupby('game_pk')\
['pitch_number'].transform(lambda x: (x == x.max()).astype(int))

# last pitch per at bat
df['final_pitch_at_bat'] = df.groupby(['game_pk', 'at_bat_number', 'inning_topbot'])\
['pitch_number'].transform(lambda x: (x == x.max()).astype(int))

# last pitch per inning_topbot
df['final_pitch_inning'] = 0
df.loc[(df['final_pitch_at_bat'] == 1) & (df['inning_topbot'] \
!= df.groupby(['game_pk', 'inning_topbot'])['inning_topbot'].shift(-1)), \
'final_pitch_inning'] = 1
df['final_pitch_inning'].fillna(1, inplace=True)

# Calculate the score resulting from each pitch.
df['runs_score_on_pitch'] = df['des'].str.count("scores")
df.loc[df['events'] == 'home_run', 'runs_score_on_pitch'] += 1
df['bat_score_after'] = df['bat_score'] + df['runs_score_on_pitch']

# Calculate accumlated socres per innings
df['score_start_inning'] = df.groupby(['gmae_pk', 'inning', 'inning_topbot'])['bat_score'].transform('minimun')
df['score_end_inning'] = df.groupby(['gmae_pk', 'inning', 'inning_topbot'])['bat_score'].transform('maximum')
df['cum_runs_in_inning'] = df.groupby(['game_pk', 'inning', 'inning_topbot']).cumsum()['runs_score_on_pitch']
df['runs_to_end_inning'] = df['score_end_inning'] - df['score']

df['base_out_state'] = df['outs_when_up'].astype(str) + " outs, " + \
df['on_1b'].notna().replace({True: "1b", False: "_"}) + ", " + \
df['on_2b'].notna().replace({True: "2b", False: "_"}) + ", " + \
df['on_3b'].notna().replace({True: "3b", False: "_"})

re_table = df.groupby('base_out_state')['runs_score_on_pitch'].mean().reset_index(name='avg_runs')

print(re_table)
#%%

# 無效出局
single_outs = ["strikeout", "caught_stealing_2b", "pickoff_caught_stealing_2b",
"other_out", "caught_stealing_3b", "caught_stealing_home",
"field_out", "force_out", "pickoff_1b", "batter_interference",
"fielders_choice", "pickoff_2b", "pickoff_caught_stealing_3b",
"pickoff_caught_stealing_home"
]
single_outs_df = df[df['events'].isin(single_outs)]

# 排序
df.sort_values(by = ['game_pk', 'at_bat_number', 'pitch_number'], inplace=True)

# last pitch per game
df['final_pitch_game'] = df.groupby('game_pk')\
['pitch_number'].transform(lambda x: (x == x.max()).astype(int))

# last pitch per at bat
df['final_pitch_at_bat'] = df.groupby(['game_pk', 'at_bat_number', 'inning_topbot'])\
['pitch_number'].transform(lambda x: (x == x.max()).astype(int))

# last pitch per inning_topbot
df['final_pitch_inning'] = 0
df.loc[df['final_pitch_at_bat'] == 1 & df['inning_topbot'] != df.groupby(['game_pk', 'inning_topbot'])['inning_topbot'].shift(-1), 'final_pitch_inning'] = 1
df['final_pitch_inning'].fillna(1, inplace=True)

# Calculate the score resulting from each pitch.
df['runs_score_on_pitch'] = df['des'].str.count("scores")
df.loc[df['events'] == 'home_run', 'runs_score_on_pitch'] += 1
df['bat_score_after'] = df['bat_score'] + df['runs_score_on_pitch']

# Calculate accumlated socres per innings
df['score_start_inning'] = df.groupby(['gmae_pk', 'inning', 'inning_topbot'])['bat_score'].transform('minimun')
df['score_end_inning'] = df.groupby(['gmae_pk', 'inning', 'inning_topbot'])['bat_score'].transform('maximum')
df['cum_runs_in_inning'] = df.groupby(['game_pk', 'inning', 'inning_topbot']).cumsum()['runs_score_on_pitch']
df['runs_to_end_inning'] = df['score_end_inning'] - df['score']

df['base_out_state'] = df['outs_when_up'].astype(str) + " outs, " + \
df['on_1b'].notna().replace({True: "1b", False: "_"}) + ", " + \
df['on_2b'].notna().replace({True: "2b", False: "_"}) + ", " + \
df['on_3b'].notna().replace({True: "3b", False: "_"})

re_table = df.groupby('base_out_state')['runs_score_on_pitch'].mean().reset_index(name='avg_runs')
    
    return re_table
#%%