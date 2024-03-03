#pip --version # 檢查使用版本
pip install pybaseball
pip3 install pybaseball # 如果檢查到的版本是使用 2.x 版的 Python
pip install seaborn
# %%
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
from pybaseball import statcast_batter
import pandas as pd

merged_df = pd.DataFrame()

# Load your dataset
df = pd.read_csv(r"C:\Users\閻天立\Downloads\2022_batter_oppo.csv")

# Fill NaN values with 0
df.fillna(0, inplace=True)

id = df["player_id"].tolist()

# List to store DataFrames for each player
dfs = []

# Iterate over each player ID and fetch data
for player_id in id:
    pitcher_data = statcast_pitcher(
        start_dt="2022-04-07", end_dt="2022-11-05",
        player_id=player_id)
    dfs.append(pitcher_data)

# Concatenate the DataFrames along the rows 
merged_df = pd.concat(dfs, axis=0)

import pandas as pd
merged_df.to_csv(r"C:\Users\閻天立\Desktop\pybaseball\2022_batter_oppo.csv")



#%%








#%%
pitcher.to_csv(r"C:\Users\user\Desktop\baseballdata\pitcher23.csv")
# %%
from pybaseball import statcast_pitcher_exitvelo_barrels

# 獲得 2023 球季投手投球被擊出的打擊初速相關數據
df = statcast_pitcher_exitvelo_barrels(2023)

df.to_csv(r"C:\Users\user\Desktop\baseballdata\pitcher23.csv")


# %%