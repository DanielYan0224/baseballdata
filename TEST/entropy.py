#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import entropy

# Define Discrete Entropy
def DE(data, bins_edge):
    hist, _ = np.histogram(data, bins=bins_edge)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]

    return entropy(prob, base=2)

###########################################
# Import Data
file_path = r"C:\Users\user\Desktop\baseballdata\data2022.csv" 
df = pd.read_csv(file_path)
###########################################
# Cut the Data
bins_edge = [0, 70, 90, 100, 110, 120]
df['launch_speed'] = df['launch_speed'].fillna(0)
df['launch_speed_histro'] = pd.cut(df['launch_speed'], bins=bins_edge, right=False)

# 把切割的區間轉成字串顯示在histrogram
df['launch_speed_histro'] = df['launch_speed_histro'].astype(str)

print(DE(df['launch_speed'], bins_edge))

# Plot histrogram
fig = px.histogram(df, x='launch_speed_histro', \
histnorm='probability density',\
title='Histogram of Launch Speed Ranges', text_auto=True)
fig.write_html('histrogram.html', auto_open=True)
#%%