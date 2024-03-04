#%%
from pybaseball import statcast_pitcher
import pandas as pd
import numpy as np

data = statcast_pitcher\
(start_dt="2018-03-29", end_dt="2018-11-01", player_id=543037)

data.to_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2018.csv")
#%%