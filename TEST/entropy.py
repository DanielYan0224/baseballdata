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
df = df[(df['description'] == 'hit_into_play') &
        (df['events'] != 'field_out')]

#x = 'launch_angle'
x = 'launch_speed'
data = df[x].dropna()

# plot histogram without sorting
# fig = px.histogram(df[x], x=x, histnorm='probability density',\
# title= f'Histogram of {x} Ranges', text_auto=True)
# fig.write_html('histrogram.html', auto_open=True)


###########################################
# Cut the Data
#bins = [data.min(), -20, 0, 30, 50, data.max()]
bins = [data.min(), 70, 80, 100, data.max()]

###########################################
labels = [f'[{bins[i]}, {bins[i+1]})' \
        for i in range(len(bins)-1)]

data_cut = pd.cut(data, bins=bins, right=False, labels=labels).dropna()
df['intervals'] = data_cut

###########################################
# Plot histrogram
fig = px.histogram(df, x='intervals', histnorm='percent',
            title=f'Histogram of {x} Ranges',
            category_orders={'intervals': labels},
            text_auto=True)
fig.show()
fig.write_html('hisstrogram.html', auto_open=True)
###########################################
print(DE(data, bins))
#%% 
