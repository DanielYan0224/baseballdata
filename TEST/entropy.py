#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import entropy

# Define Discrete Entropy
def DE(data_x, bins_x):
    hist, _ = np.histogram(data_x, bins=bins_x)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]

    return entropy(prob, base=2)

# H(X,Y)
def JE(data_x, data_y, bins_x, bins_y):
    hist, _, _ = np.histogram2d(data_x, data_y, bins=[bins_x, bins_y])
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]

    return entropy(prob.flatten(), base=2)

#H(Y|X)=H(X,Y)-H(X)
def CE(data_x, data_y, bins_x, bins_y):
    conditional_entropy = \
    JE(data_x, data_y, bins_x, bins_y) - DE(data_x, bins_x)

    return conditional_entropy
###########################################
# Import Data
file_path = r"C:\Users\user\Desktop\baseballdata\data2022.csv" 
df = pd.read_csv(file_path)

x = 'launch_speed'
y = 'launch_angle'
data_x = df[x]
data_y = df[y]

fig = px.histogram(df[x], x=x, histnorm='probability density',\
title='Histogram of Launch Speed Ranges', text_auto=True)
fig.write_html('histrogram.html', auto_open=True)
print(DE(data_x, bins_x))
print(DE(df['launch_angle'], bins_x))

###########################################
# Cut the Data
bins_x = [0, 70, 90, 100, 110, 120]
bins_y = [-80, -20, -10, 0, 40, 90]

###########################################
df['launch_speed'] = df['launch_speed'].fillna(0)
df[f'{x}_histro'] = pd.cut(df[x], \
bins=bins_x, right=False)

print(df[f'{x}_histro'], df[x])

# 把切割的區間轉成字串顯示在histrogram
df[f'{x}_histro'] = df[f'{x}_histro'].astype(str)

# Plot histrogram
fig = px.histogram(df, x=f'{x}_histro', \
histnorm='probability density',\
title='Histogram of Launch Speed Ranges', text_auto=True)
fig.write_html('hisstrogram.html', auto_open=True)
###########################################
#%% 