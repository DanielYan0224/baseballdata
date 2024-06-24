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

#################################
#整理資料
def extract_year(file_path):
    match = re.search(r'\d{4}', file_path)
    if match:
        return int(match.group())  # Convert to integer
    else:
        return None 
    
def load_dataframe(file_path):
    df = pd.read_csv(file_path)
    df.insert(loc=1, column='year', value=extract_year(file_path))
    return df

# 只留下安打與出局
def new_event(events):
    mapping = {'single': "single", 'double': "double", 
            'triple': "triple", 'home_run': "home_run", 
            'field_out': "field_out", "double_play": "DP"}
    return mapping.get(events, "others")  # 其他的都是others

def weight(events):
    mapping = mapp
    return mapping.get(events, np.nan)  # 其他的都是nan

def sort_data (df):
    df = df.dropna(subset=[x_label, y_label], how='any')
    df = df[df['description'] == 'hit_into_play']
    df['new_event'] = df['events'].apply(new_event).astype(str)
    return df

def grill(df, x_lines, y_lines):
    for i in range(len(x_lines) - 1):  
        for j in range(len(y_lines) - 1): 
            label = f'{len(y_lines) - 2 -j}_{i}'
            df.loc[
                (df[x_label] >= x_lines[i]) & \
                (df[x_label] < x_lines[i+1]) & \
                (df[y_label] >= y_lines[j]) & \
                (df[y_label] < y_lines[j+1]), "grill"
            ] =  label
    return df


file_path_1 = r"C:\Users\user\Desktop\baseballdata\2022sppitchingdata.csv"
file_path_2 = r"C:\Users\user\Desktop\baseballdata\2023sppitchingdata.csv"

x_label = 'launch_angle'
y_label = 'launch_speed'

df1 = sort_data(load_dataframe(file_path_1))

df2 = sort_data(load_dataframe(file_path_2))

df = pd.concat([df1, df2], axis=0)

#################################
#可調整
year = 2022
file_path = file_path_1
study_df = load_dataframe(file_path)
x_grids = 40
y_grids = 40
mapp = {'single': 1, 
'double': 2, 
'triple': 3, 
'home_run': 4,
'DP': 0, 
'field_out': 0,
        }
#################################

# Draw the scatter plot

# 定義grid
x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = 0, 120

grid_size_x = (x_max - x_min) / x_grids
grid_size_y = (y_max - y_min) / y_grids
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)


df2022 = grill(sort_data(df1), x_lines, y_lines)
df2022["weighted_events"] = df2022["new_event"].apply(weight)

pitch_type = 'FF'
df2022 = df2022[df2022['pitch_type']==pitch_type]

def matrix_year(df, mapp):
    numerator = df.groupby('grill')['weighted_events'].sum().dropna().to_dict()
    denominator = df.groupby('grill').size().dropna().to_dict()
    
    ratio = {}
    for grill in denominator:
        if grill == 'nan':
            continue
        if grill in numerator:
            if denominator[grill] > 0:
                prob = numerator[grill] / denominator[grill]
                ratio[grill] = prob
            
    A = [[-2 for _ in range(len(x_lines) - 1)] for _ in range(len(y_lines) - 1)]
    for key in ratio:
        parts = key.split('_')
        if len(parts) == 2:
            j, i = [int(part) for part in parts]
            A[j][i] = ratio[key]

    return A



# normailze array 
def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

data = np.array(matrix_year(df2022, mapp))

# 求矩陣數值
flattened_data = [value for sublist in data 
                for value in sublist if value != -2]


filtered_data = data
 


# 把初速小於60的全部設定成 0, zone 1
mask_1 = filtered_data[22:40, :] >= 0
filtered_data[22:40, :][mask_1]= 0.2

original_data = filtered_data.copy()

# 設定 zone 2
mask_2 = (original_data[0:22, 1:26] >= 0) & \
    (original_data[0:22, 1:26] <= 0.3)
filtered_data[0:22, 1:26][mask_2] = 0.4

# #設定 region 3
mask_3 = (original_data[0:22, 25:40] >= 0) & \
    (original_data[0:22, 25:40] <= 0.3)
filtered_data[0:22, 25:40][mask_3] = 0.6

# 設定 region 4
mask_4 = (original_data[1:22,18:30] >= 0.3) & \
    (original_data[1:22, 18:30] <= 1)
filtered_data[1:22, 18:30][mask_4] = 0.8

# 設定 region 5
mask_5 = (original_data[0:8,23:31] >= 1) 
filtered_data[0:8, 23:31][mask_5] = 1


# 微調heat map
mask_6 = filtered_data[0:22, 0:19] >= 0
filtered_data[0:22, 0:19][mask_6] = 0.4

filtered_data[4,30] = 0.2

# red 
filtered_data[17,22] = filtered_data[21,25] = filtered_data[21,27]=\
filtered_data[14,24] = filtered_data[20,29] = filtered_data[20,26] = \
filtered_data[21,28] = filtered_data[19,28] = filtered_data[21,29] = 0.8
filtered_data[20,22] = 0.8

# yellow
filtered_data[15,19] = filtered_data[14,20] = 0.4

# orange
filtered_data[19,30] = filtered_data[19,31] = filtered_data[18,31] = \
filtered_data[17,30] = filtered_data[16,31] = filtered_data[16,32] = \
filtered_data[5,30] = filtered_data[22,30] = \
filtered_data[21,30] = filtered_data[16,30] = filtered_data[17,31] = \
filtered_data[15,32] = 0.6

# black
filtered_data[18,26] = filtered_data[18,25] = \
filtered_data[17,25] = filtered_data[17,24] = \
filtered_data[16,24] = filtered_data[15,24] = \
filtered_data[14,23] = filtered_data[13,23] = \
filtered_data[12,23] = filtered_data[11,23] = \
filtered_data[16,25] = filtered_data[13,24] = \
filtered_data[14,24] = 0.75

# purple
filtered_data[2,22] = filtered_data[3,22] = filtered_data[4,22] = \
filtered_data[5,22] = filtered_data[6,22] = filtered_data[1,22] = \
filtered_data[7,22] =1
##############
fighp = go.Figure()

colorscale = [
    [0.0, 'blue'],   # 0
    [0.2, 'green'],  # 0.2
    [0.4, 'yellow'], # 0.4
    [0.6, 'orange'],
    [0.75, 'black'], # 0.6
    [0.8, 'red'],    # 0.8
    [1.0, 'purple']  # 1.0
]


fighp = go.Figure(data=go.Heatmap(
    z=filtered_data, 
    x=x_lines,  # x 邊界
    y=y_lines[::-1],  # y 邊界
    zmin=0,
    zmax=1,
    colorscale=colorscale,  
    hoverongaps=False,  
    hoverinfo='z',   
    # texttemplate="%{text}<br>%{z}",  
    colorbar=dict(title='Data Values')
))

fighp.update_layout(
    title=pitch_type
)

# 添加x y 格線
for x in x_lines:
    fighp.add_shape(type='line', x0=x ,y0=y_min, x1=x, y1=y_max,
            xref='x', yref='y',
            line=dict(color="Gray", width=2))

for y in y_lines:
    fighp.add_shape(type='line', x0=x_min ,y0=y, x1=x_max, y1=y,
            xref='x', yref='y',
            line=dict(color="Gray", width=2))

fighp.update_layout(
    coloraxis_colorbar=dict(
        x=1.2,  # Move color bar further to the right
        title='Color Scale'  # Title for the color bar
    ),
    legend=dict(
        x=-0.2,  # Move legend slightly to the right
        y=0.5,   # Center legend vertically
        traceorder='normal',
        bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: add a background color for better readability
        bordercolor='Black',
        borderwidth=1
    )
)

fighp.show()

def find_value(matrix, target_value):
    Y_zones = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value == target_value:
                Y_zones.append(f'{i}_{j}')
    return Y_zones

Y_1 = find_value(data, 0.2)
Y_2 = find_value(data, 0.4)
Y_3 = find_value(data, 0.6)
Y_4 = find_value(data, 0.75)
Y_5 = find_value(data, 0.8)
Y_6 = find_value(data, 1)

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
    return zone_mapping.get(grill, 0)  # Default to 0 if not found

# Apply the classification to df2022
df2022['Y_zone'] = df2022['grill'].apply(classify_grill)

# print(df2022[df2022['Y_zone']==0]['grill'])

counts =  df2022['Y_zone'].value_counts()
proportions = df2022['Y_zone'].value_counts(normalize=True)
shannon_entropy = -np.sum(proportions * np.log2(proportions))

result = pd.DataFrame({'Count': counts, 'Proportion': proportions, 
                       "Entropy": shannon_entropy})
print(result)
#%%