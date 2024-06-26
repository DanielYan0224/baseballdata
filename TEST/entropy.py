#%%
import re
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from fractions import Fraction as F
from math import log
from plotly.graph_objs import Figure
import dash
from dash import Dash, dcc, html, Input, Output, State 
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.ndimage import median_filter, gaussian_filter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path
from plotly.subplots import make_subplots
from scipy.stats import entropy
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform

file_1 = r"C:\Users\user\Desktop\baseballdata\cole1317_pirates.csv"
file_2 = r"C:\Users\user\Desktop\baseballdata\cole1819_cheater.csv"
file_3 = r"C:\Users\user\Desktop\baseballdata\cole2023_yankee.csv"

df1 = pd.read_csv(file_1)
df1['team'] = 'pirate'

df2 = pd.read_csv(file_2)
df2['team'] = 'astro'

df3 = pd.read_csv(file_3)
df3['team'] = 'yankee'

df = pd.concat([df1, df2, df3], axis=0)

df = df[(df['description'] == 'hit_into_play') &
        ( df['pitch_type'] == 'FF')].dropna(subset=['launch_angle', 'launch_speed'])
# 建立 grill
x_min, x_max = -90, 90
y_min, y_max = 0, 120

grid_size_x = (x_max - x_min) / 40
grid_size_y = (y_max - y_min) / 40
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)


def grill(df, x_lines, y_lines):
    for i in range(len(x_lines) - 1):  
        for j in range(len(y_lines) - 1): 
            label = f'{len(y_lines) - 2 -j}_{i}'
            df.loc[
                (df['launch_angle'] >= x_lines[i]) & \
                (df['launch_angle'] < x_lines[i+1]) & \
                (df['launch_speed'] >= y_lines[j]) & \
                (df['launch_speed'] < y_lines[j+1]), "grill"
            ] =  label
    return df
df = grill(df, x_lines, y_lines)


# 定義區間
Y_1 = ['22_0', '22_1', '22_2', '22_3', '22_4', '22_5', '22_6', '22_7', '22_8', '22_9', '22_10', '22_11', '22_12', '22_13', '22_14', '22_15', '22_16', '22_17', '22_18', '22_19', '22_20', '22_21', '22_22', '22_23', '22_24', '22_25', '22_26', '22_27', '22_28', '22_29', '22_30', '22_31', '22_32', '22_33', '22_34', '22_35', '22_36', '22_37', '22_38', '22_39', '23_0', '23_1', '23_2', '23_3', '23_4', '23_5', '23_6', '23_7', '23_8', '23_9', '23_10', '23_11', '23_12', '23_13', '23_14', '23_15', '23_16', '23_17', '23_18', '23_19', '23_20', '23_21', '23_22', '23_23', '23_24', '23_25', '23_26', '23_27', '23_28', '23_29', '23_30', '23_31', '23_32', '23_33', '23_34', '23_35', '23_36', '23_37', '23_38', '24_0', '24_1', '24_2', '24_3', '24_4', '24_5', '24_6', '24_7', '24_8', '24_9', '24_10', '24_11', '24_12', '24_13', '24_14', '24_15', '24_16', '24_17', '24_18', '24_19', '24_20', '24_21', '24_22', '24_23', '24_24', '24_25', '24_26', '24_27', '24_29', '24_30', '24_32', '24_33', '24_36', '24_37', '24_39', '25_0', '25_1', '25_2', '25_3', '25_4', '25_5', '25_6', '25_7', '25_8', '25_9', '25_10', '25_11', '25_12', '25_13', '25_15', '25_16', '25_17', '25_18', '25_19', '25_20', '25_21', '25_22', '25_23', '25_24', '25_25', '25_26', '25_27', '25_29', '25_30', '25_31', '25_32', '25_33', '25_34', '25_37', '26_1', '26_2', '26_3', '26_4', '26_5', '26_6', '26_7', '26_8', '26_9', '26_10', '26_11', '26_12', '26_13', '26_14', '26_15', '26_16', '26_17', '26_18', '26_19', '26_20', '26_21', '26_22', '26_23', '26_24', '26_25', '26_27', '26_28', '26_29', '26_30', '26_31', '26_32', '26_33', '26_34', '26_35', '26_36', '26_37', '27_0', '27_1', '27_2', '27_3', '27_4', '27_5', '27_6', '27_7', '27_8', '27_9', '27_10', '27_11', '27_12', '27_13', '27_14', '27_15', '27_16', '27_17', '27_18', '27_19', '27_20', '27_21', '27_23', '27_25', '27_26', '27_27', '27_28', '27_29', '27_31', '27_35', '27_36', '27_37', '27_38', '28_2', '28_4', '28_5', '28_6', '28_7', '28_8', '28_9', '28_10', '28_11', '28_12', '28_13', '28_14', '28_15', '28_16', '28_17', '28_18', '28_19', '28_20', '28_21', '28_22', '28_24', '28_25', '28_26', '28_27', '28_28', '28_30', '28_31', '28_32', '28_33', '28_34', '28_35', '28_37', '29_1', '29_3', '29_4', '29_5', '29_6', '29_7', '29_8', '29_9', '29_10', '29_11', '29_12', '29_13', '29_14', '29_15', '29_16', '29_17', '29_18', '29_19', '29_20', '29_21', '29_22', '29_24', '29_25', '29_26', '29_27', '29_28', '29_29', '29_31', '29_34', '29_35', '29_36', '30_3', '30_4', '30_5', '30_6', '30_7', '30_8', '30_9', '30_10', '30_11', '30_12', '30_14', '30_15', '30_16', '30_17', '30_18', '30_19', '30_20', '30_21', '30_22', '30_23', '30_24', '30_25', '30_26', '30_27', '30_29', '30_30', '30_31', '30_33', '30_37', '31_2', '31_4', '31_6', '31_7', '31_8', '31_10', '31_11', '31_12', '31_14', '31_15', '31_16', '31_17', '31_18', '31_19', '31_20', '31_21', '31_22', '31_23', '31_24', '31_26', '31_27', '31_28', '31_29', '31_30', '31_34', '31_36', '32_0', '32_12', '32_14', '32_15', '32_16', '32_18', '32_19', '32_20', '32_21', '32_23', '32_27', '32_28', '32_29', '32_33', '32_37', '33_9', '33_11', '33_13', '33_17', '33_24', '33_26', '33_29', '34_18', '34_21', '34_22', '36_26']
Y_2 = ['0_17', '0_18', '1_16', '1_17', '1_18', '2_16', '2_17', '2_18', '3_12', '3_13', '3_14', '3_15', '3_16', '3_17', '3_18', '4_12', '4_13', '4_14', '4_15', '4_16', '4_17', '4_18', '5_10', '5_11', '5_12', '5_13', '5_14', '5_15', '5_16', '5_17', '5_18', '6_8', '6_9', '6_10', '6_11', '6_12', '6_13', '6_14', '6_15', '6_16', '6_17', '6_18', '7_9', '7_10', '7_11', '7_12', '7_13', '7_14', '7_15', '7_16', '7_17', '7_18', '8_7', '8_8', '8_9', '8_10', '8_11', '8_12', '8_13', '8_14', '8_15', '8_16', '8_17', '8_18', '9_8', '9_9', '9_10', '9_11', '9_12', '9_13', '9_14', '9_15', '9_16', '9_17', '9_18', '9_19', '10_5', '10_6', '10_7', '10_8', '10_9', '10_10', '10_11', '10_12', '10_13', '10_14', '10_15', '10_16', '10_17', '10_18', '10_19', '11_4', '11_5', '11_6', '11_7', '11_8', '11_9', '11_10', '11_11', '11_12', '11_13', '11_14', '11_15', '11_16', '11_17', '11_18', '11_19', '12_4', '12_5', '12_6', '12_7', '12_8', '12_9', '12_10', '12_11', '12_12', '12_13', '12_14', '12_15', '12_16', '12_17', '12_18', '12_19', '12_20', '13_4', '13_5', '13_6', '13_7', '13_8', '13_9', '13_10', '13_11', '13_12', '13_13', '13_14', '13_15', '13_16', '13_17', '13_18', '13_19', '13_20', '14_2', '14_3', '14_4', '14_5', '14_6', '14_7', '14_8', '14_9', '14_10', '14_11', '14_12', '14_13', '14_14', '14_15', '14_16', '14_17', '14_18', '14_19', '14_20', '14_21', '15_2', '15_3', '15_4', '15_5', '15_6', '15_7', '15_8', '15_9', '15_10', '15_11', '15_12', '15_13', '15_14', '15_15', '15_16', '15_17', '15_18', '15_19', '15_20', '15_21', '16_2', '16_3', '16_4', '16_5', '16_6', '16_7', '16_8', '16_9', '16_10', '16_11', '16_12', '16_13', '16_14', '16_15', '16_16', '16_17', '16_18', '16_19', '16_20', '16_21', '16_22', '17_1', '17_2', '17_3', '17_4', '17_5', '17_6', '17_7', '17_8', '17_9', '17_10', '17_11', '17_12', '17_13', '17_14', '17_15', '17_16', '17_17', '17_18', '17_19', '17_20', '17_21', '17_22', '18_1', '18_2', '18_3', '18_4', '18_5', '18_6', '18_7', '18_8', '18_9', '18_10', '18_11', '18_12', '18_13', '18_14', '18_15', '18_16', '18_17', '18_18', '18_19', '18_20', '18_21', '18_22', '19_1', '19_2', '19_3', '19_4', '19_5', '19_6', '19_7', '19_8', '19_9', '19_10', '19_11', '19_12', '19_13', '19_14', '19_15', '19_16', '19_17', '19_18', '19_19', '19_20', '19_21', '19_22', '19_23', '20_1', '20_2', '20_3', '20_4', '20_5', '20_6', '20_7', '20_8', '20_9', '20_10', '20_11', '20_12', '20_13', '20_14', '20_15', '20_16', '20_17', '20_18', '20_19', '20_20', '20_21', '20_22', '20_23', '21_0', '21_1', '21_2', '21_3', '21_4', '21_5', '21_6', '21_7', '21_8', '21_9', '21_10', '21_11', '21_12', '21_13', '21_14', '21_15', '21_16', '21_17', '21_18', '21_19', '21_20', '21_21', '21_22', '21_23']
Y_3 = ['3_31', '3_32', '4_31', '4_32', '4_33', '4_34', '4_36', '5_30', '5_31', '5_32', '5_33', '5_34', '5_35', '6_29', '6_30', '6_31', '6_32', '6_33', '6_34', '6_35', '6_36', '7_29', '7_30', '7_31', '7_32', '7_33', '7_34', '7_35', '7_36', '7_37', '8_27', '8_28', '8_29', '8_30', '8_31', '8_32', '8_33', '8_34', '8_35', '8_36', '8_37', '8_38', '9_25', '9_26', '9_27', '9_28', '9_29', '9_30', '9_31', '9_32', '9_33', '9_34', '9_35', '9_36', '9_37', '9_38', '9_39', '10_25', '10_26', '10_27', '10_28', '10_29', '10_30', '10_31', '10_32', '10_33', '10_34', '10_35', '10_36', '10_37', '10_38', '10_39', '11_25', '11_26', '11_27', '11_28', '11_29', '11_30', '11_31', '11_32', '11_33', '11_34', '11_35', '11_36', '11_37', '11_38', '11_39', '12_26', '12_27', '12_28', '12_29', '12_30', '12_31', '12_32', '12_33', '12_34', '12_35', '12_36', '12_37', '12_38', '12_39', '13_26', '13_27', '13_28', '13_29', '13_30', '13_31', '13_32', '13_33', '13_34', '13_35', '13_36', '13_37', '13_38', '13_39', '14_27', '14_28', '14_29', '14_30', '14_31', '14_32', '14_33', '14_34', '14_35', '14_36', '14_37', '14_38', '14_39', '15_29', '15_30', '15_31', '15_32', '15_33', '15_34', '15_35', '15_36', '15_37', '15_38', '15_39', '16_30', '16_31', '16_32', '16_33', '16_34', '16_35', '16_36', '16_37', '16_38', '16_39', '17_30', '17_31', '17_32', '17_33', '17_34', '17_35', '17_36', '17_37', '17_38', '17_39', '18_30', '18_31', '18_32', '18_33', '18_34', '18_35', '18_36', '18_37', '18_38', '18_39', '19_30', '19_31', '19_32', '19_33', '19_34', '19_35', '19_36', '19_37', '19_38', '19_39', '20_30', '20_31', '20_32', '20_33', '20_34', '20_35', '20_36', '20_37', '21_30', '21_31', '21_32', '21_33', '21_34', '21_35', '21_36', '21_37', '21_38']
Y_4 = ['11_23', '12_23', '13_23', '13_24', '14_23', '14_24', '15_24', '16_24', '16_25', '17_24', '17_25', '18_25', '18_26']
Y_5 = ['1_19', '1_20', '1_21', '2_19', '2_20', '3_19', '3_20', '3_21', '4_19', '4_20', '4_21', '4_30', '5_19', '5_20', '5_21', '5_23', '5_29', '6_19', '6_20', '6_21', '6_23', '6_24', '6_28', '7_19', '7_20', '7_21', '7_23', '7_24', '7_25', '7_27', '7_28', '8_19', '8_20', '8_21', '8_22', '8_23', '8_24', '8_26', '9_20', '9_21', '9_22', '9_23', '9_24', '10_20', '10_21', '10_22', '10_23', '10_24', '11_20', '11_21', '11_22', '11_24', '12_21', '12_22', '12_24', '12_25', '13_21', '13_22', '13_25', '14_22', '14_25', '14_26', '15_22', '15_23', '15_25', '15_26', '15_27', '15_28', '16_23', '16_26', '16_27', '16_28', '16_29', '17_23', '17_26', '17_27', '17_28', '17_29', '18_23', '18_24', '18_27', '18_28', '18_29', '19_24', '19_25', '19_26', '19_27', '19_28', '19_29', '20_24', '20_25', '20_26', '20_27', '20_28', '20_29', '21_24', '21_25', '21_26', '21_27', '21_28', '21_29']
Y_6 = ['0_19', '0_20', '0_21', '0_22', '0_23', '0_24', '0_25', '0_26', '1_22', '1_23', '1_24', '1_25', '1_26', '1_27', '2_21', '2_22', '2_23', '2_24', '2_25', '2_26', '2_27', '2_28', '2_30', '3_22', '3_23', '3_24', '3_25', '3_26', '3_27', '3_28', '3_29', '3_30', '4_22', '4_23', '4_24', '4_25', '4_26', '4_27', '4_28', '4_29', '5_22', '5_24', '5_25', '5_26', '5_27', '5_28', '6_22', '6_25', '6_26', '6_27', '7_22', '7_26', '8_25']

# Create mapping dictionary
zone_mapping = {}

# Map each Y zone to its corresponding numeric value
for zone in Y_1:
    zone_mapping[zone] = 1
for zone in Y_2:
    zone_mapping[zone] = 2
for zone in Y_3:
    zone_mapping[zone] = 3
for zone in Y_4:
    zone_mapping[zone] = 4
for zone in Y_5:
    zone_mapping[zone] = 5
for zone in Y_6:
    zone_mapping[zone] = 6

# Function to classify 'grill' based on the mapping dictionary
def classify_grill(grill):
    return zone_mapping.get(grill, 0)

df['Y_zone'] = df['grill'].apply(classify_grill).dropna()

### 繪製scatter plot 

fig_scatter = px.scatter(df, x='launch_angle', y="launch_speed",
                         color = 'Y_zone')
fig_scatter.show()

### 繪製dendrogram

df_den = df[(df['pitch_type'] == 'FF') & (df['description'] == 'hit_into_play')]
# 假設 df1 是你已經加載的 DataFrame
df_dendrogram = df_den[['release_speed', 'release_pos_x', 'release_pos_z',
                     'hit_location', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                     'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 
                     'az', 'sz_top', 'sz_bot', 'effective_speed', 
                    'release_extension', 'spin_axis', 'Y_zone']].dropna()

X = df_dendrogram.T

# create upper tree map 
fig = ff.create_dendrogram(X, orientation='bottom', labels=X.index.tolist())
for i in range(len(fig['data'])):
    fig['data'][i]['yaxis'] = 'y2'

# create tree map 
dendro_side = ff.create_dendrogram(X, orientation='right')
for i in range(len(dendro_side['data'])):
    dendro_side['data'][i]['xaxis'] = 'x2'

# add tree map
for data in dendro_side['data']:
    fig.add_trace(data) 

# create heat map 
dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
dendro_leaves = list(map(int, dendro_leaves))
data_dist = pdist(X)
heat_data = squareform(data_dist)
heat_data = heat_data[dendro_leaves, :]
heat_data = heat_data[:, dendro_leaves] / 1000

heatmap = [
    go.Heatmap(
        x=dendro_leaves,
        y=dendro_leaves,
        z=heat_data,
        colorscale='Blues'
    )
]

heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

for data in heatmap:
    fig.add_trace(data)

# layout
fig.update_layout({
    'width': 800,
    'height': 800,
    'showlegend': False,
    'hovermode': 'closest'
})

# xaxis
fig.update_layout(xaxis={
    'domain': [.15, 1],
    'mirror': False,
    'showgrid': False,
    'showline': False,
    'zeroline': False,
    'ticks': ""
})

# xaxis2
fig.update_layout(xaxis2={
    'domain': [0, .15],
    'mirror': False,
    'showgrid': False,
    'showline': False,
    'zeroline': False,
    'showticklabels': False,
    'ticks': ""
})

# yaxis
fig.update_layout(yaxis={
    'domain': [0, .85],
    'mirror': False,
    'showgrid': False,
    'showline': False,
    'zeroline': False,
    'showticklabels': False,
    'ticks': ""
})

# yaxis2
fig.update_layout(yaxis2={
    'domain': [.825, .975],
    'mirror': False,
    'showgrid': False,
    'showline': False,
    'zeroline': False,
    'showticklabels': False,
    'ticks': ""
})

# fig.show()
# 計算entropy
covar_x = ['release_speed', 'release_pos_x', 'release_pos_z',
                     'hit_location', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                     'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 
                     'az', 'sz_top', 'sz_bot', 'effective_speed', 
                    'release_extension', 'spin_axis', 'Y_zone']

df_covar = df[covar_x].dropna()

### 計算 entropy

# step1 計算Y (Y_zone) 的entropy
y_counts = df['Y_zone'].value_counts()
y_prob = y_counts / len(df)
y_entropy = entropy(y_prob)

conditional_entropies = {}
drop_entropies = {}
for feature in covar_x:
    # 轉資料型態並用中位數填充nan   
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    df[feature].fillna(df[feature].median(), inplace=True)

    # 離散化
    df[f'{feature}_bin'] = pd.cut(df[feature], bins=10, labels=False)
    
    # 遍歷covar set
    conditional_entropy = 0
    for feature_bin in df[f'{feature}_bin'].unique():
        subset = df[df[f'{feature}_bin'] == feature_bin]
        if len(subset) > 0:
            y_zone_counts_given_feature = subset['Y_zone'].value_counts()
            y_zone_prob_given_feature = y_zone_counts_given_feature / len(subset)
            subset_entropy = entropy(y_zone_prob_given_feature)
            weight = len(subset) / len(df)
            conditional_entropy += weight * subset_entropy
    
    conditional_entropies[feature] = conditional_entropy
    drop_entropies[feature] = y_entropy - conditional_entropy

results_df = pd.DataFrame({
    'Conditional Entropies': conditional_entropies,
    'Drop Entropies': drop_entropies
})

# 按照 DE 排列
results_df = results_df.sort_values(by='Drop Entropies', ascending=False)


print(results_df)
# print(f"Entropy of Y_zone: {y_entropy}")
# for feature, cond_entropy in conditional_entropies.items():
#     print(f"Conditional Entropy of Y_zone given {feature}: {cond_entropy}")

# for feature, cond_entropy in drop_entropies.items():    
#     print(f"drop CE of Y_zone given {feature}: {cond_entropy:.5f}")
#%%
