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

symbol_map = {
    'single': 'circle',           
    'double': 'square',           
    'triple': 'diamond',          
    'home_run': 'star',           
    'field_out': 'x',             
    'DP': "hexagon", 
    'others': 'triangle-up'       
}

color_discrete_map = {
    'single': 'blue',
    'double': 'orange',
    'triple': 'green',
    'home_run': 'red',
    'field_out': 'purple',
    'DP': 'black',    
    'others': 'grey'  
}             



#################################
# Draw the scatter plot

# 定義grid
x_min, x_max = df[x_label].min(), df[x_label].max()
y_min, y_max = df[y_label].min(), df[y_label].max()

grid_size_x = (x_max - x_min) / x_grids
grid_size_y = (y_max - y_min) / y_grids
x_lines = np.arange(x_min, x_max, grid_size_x).tolist()
y_lines = np.arange(y_min, y_max, grid_size_y).tolist()

x_lines.append(x_max)
y_lines.append(y_max)


df2022 = grill(sort_data(df1), x_lines, y_lines)
df2022["weighted_events"] = df2022["new_event"].apply(weight)
df2023 = grill(sort_data(df2), x_lines, y_lines)
df2023["weighted_events"] = df2023["new_event"].apply(weight)

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
            
    A = [[-1 for _ in range(len(x_lines) - 1)] for _ in range(len(y_lines) - 1)]
    for key in ratio:
        parts = key.split('_')
        if len(parts) == 2:
            j, i = [int(part) for part in parts]
            A[j][i] = ratio[key]

    return A



color_list = df2022['new_event'].map(color_discrete_map).tolist()
symbol_list = df2022['new_event'].map(symbol_map).tolist()
color_list = df2023['new_event'].map(color_discrete_map).tolist()
symbol_list = df2023['new_event'].map(symbol_map).tolist()


data = np.array(matrix_year(df2022, mapp))
filtered_data = data
#gaussian_filter(data, sigma=1.5)


# for row in filtered_data:
#     print(" ".join(f"{elem:.2f}" for elem in row))

fighp = go.Figure()


##############




custom_colorscale = [
    [0, 'blue'],    
    [0.1, 'green'], 
    [0.175, 'yellow'], 
    [0.2, 'orange'],
    [1, 'red']       
]

fighp = go.Figure(data=go.Heatmap(
    z=filtered_data, 
    x=x_lines,  # x 邊界
    y=y_lines[::-1],  # y 邊界
    zmin=-1,
    zmax=3,
    colorscale='Jet',  
    hoverongaps=False,  
    hoverinfo='z',  
    colorbar=dict(title='Data Values')  
))

# 添加x y 格線
for x in x_lines:
    fighp.add_shape(type='line', x0=x ,y0=y_min, x1=x, y1=y_max,
            xref='x', yref='y',
            line=dict(color="Gray", width=2))

for y in y_lines:
    fighp.add_shape(type='line', x0=x_min ,y0=y, x1=x_max, y1=y,
            xref='x', yref='y',
            line=dict(color="Gray", width=2))

data = {}
points = {}
hulls = {}

def compute_convex_hull(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    # Closing the loop for the convex hull
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)
    return hull_points

# Add points
data1 = df2022[df2022["launch_speed_angle"]==1]
points1 = data1[[x_label, y_label]].to_numpy()

data2 = df2022[df2022["launch_speed_angle"]==2]
points2 = data2[[x_label, y_label]].to_numpy()

data3 = df2022[df2022["launch_speed_angle"]==3]
points3 = data3[[x_label, y_label]].to_numpy()

data4 = df2022[df2022["launch_speed_angle"]==4]
points4 = data4[[x_label, y_label]].to_numpy()

data5 = df2022[df2022["launch_speed_angle"]==5]
points5 = data5[[x_label, y_label]].to_numpy()

data6 = df2022[df2022["launch_speed_angle"]==6]
points6 = data6[[x_label, y_label]].to_numpy()


# Create hulls
hull_points1 = compute_convex_hull(points1)
hull_trace1 = go.Scatter(x=hull_points1[:, 0], y=hull_points1[:, 1],
                         mode='lines',
                         line=dict(width=4)
                         #name='Convex Hull 1'
                         )

hull_points2 = compute_convex_hull(points2)
hull_trace2 = go.Scatter(x=hull_points2[:, 0], 
                         y=hull_points2[:, 1],
                         mode='lines',
                         line=dict(width=4))

hull_points3 = compute_convex_hull(points3)
hull_trace3 = go.Scatter(x=hull_points3[:, 0], 
                         y=hull_points3[:, 1],
                         mode='lines',
                         line=dict(width=4))

hull_points4 = compute_convex_hull(points4)
hull_trace4 = go.Scatter(x=hull_points4[:, 0], 
                         y=hull_points4[:, 1],
                         mode='lines',
                         line=dict(width=4))

hull_points5 = compute_convex_hull(points5)
hull_trace5 = go.Scatter(x=hull_points5[:, 0], 
                         y=hull_points5[:, 1],
                         mode='lines',
                         line=dict(width=4, color="black"))

hull_points6 = compute_convex_hull(points6)
hull_trace6 = go.Scatter(x=hull_points6[:, 0], 
                         y=hull_points6[:, 1],
                         mode='lines',
                         line=dict(width=4, color="white"))

fighp.update_layout(
    title_text = year
)



fighp.add_trace(hull_trace1)
fighp.add_trace(hull_trace2)
fighp.add_trace(hull_trace3)
fighp.add_trace(hull_trace4)
fighp.add_trace(hull_trace5)
fighp.add_trace(hull_trace6)

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
#fighp.write_html("heatmp.html", auto_open=True)
#%%
# fighp = px.imshow(filtered_data, 
#             #color_continuous_scale=px.colors.sequential.Blues
#             )

# fighp.add_trace(go.Contour(
#     z=filtered_data,
#     contours=dict(
#             coloring='heatmap',
#             showlabels=True, 
#             labelfont=dict(
#                 size=12,
#                 color="white"
#             )
#         ),
#         line_width=2
#     )
# )

fighp = go.Figure(
    go.Contour(
        z=filtered_data,
        x=[i for i in range(filtered_data.shape[1])],
        y=[i for i in range(filtered_data.shape[0])],
        contours=dict(
            coloring='heatmap',
            showlabels=True, 
            labelfont=dict(
                size=12,
                color="white"
            )
        ),
        line_width=2,
        colorscale='Portland' 
    )
#     counters = dict(
#         start=filtered_data.max(),
#         end=filtered_data.min(),
#         size = (filtered_data.max() - filtered_data.min()) / 10
#     )
# 
)
# 反轉 y 軸
fighp.update_layout(
    yaxis=dict(
        autorange="reversed"  
    )
)

# 更新x y軸座標標籤
fighp.update_xaxes(tickvals=list(range(0, len(x_lines), 5)),
    ticktext =[x_lines[k] for k in range(0, len(x_lines), 5)],
    tickangle=15,
    tickfont=dict(color="black"),
    ticks="outside",tickwidth=2,
    tickcolor="crimson",
    ticklen=10)
fighp.update_yaxes(tickvals=list(range(0, len(y_lines), 5)),
    ticktext =[round(y_lines[k], 2) for k in reversed(range(0, len(y_lines), 5))],
    tickfont=dict(color="black"),
    ticks="outside",tickwidth=2,
    tickcolor="crimson",
    ticklen=10)

# 新增年份
fighp.add_annotation(
    text="2023",
    align='left',
    showarrow=False,
    xref='paper',
    yref='paper',
    x=0.5,  
    y=-0.2,  
    font=dict(
        size=20,
        color="black"
    )
)

# fighp.show()
fighp.write_html('heatmap.html', auto_open=True)
#%%
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div([
            # 顯示mapp的標籤
            html.Label(f"{key}:", style={'margin-right': '10px'}),  
            dcc.Input(
                # 輸入框的 ID
                id=f'input-{key}',
                # 輸入數字  
                type='number',
                # 默認為原本mapp的數字  
                value=value,  
                # 輸入框的外邊界
                style={'margin': '5px'} 
            )
            # 設置樣式 水平排列
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'})  
        for key, value in mapp.items() 
    ]),
    # 新增trigger
    html.Button('Update Heatmap', id='update-button', n_clicks=0), 
    dcc.Dropdown(
        # 下拉列表
        id='dataset-dropdown',  
        options=[
            {'label': '2022 Data', 'value': '2022'}, 
            {'label': '2023 Data', 'value': '2023'}  
        ],
        value='2022'  
    ),
    dcc.Graph(id='heat-map') 
])


@app.callback(
    Output('heat-map', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('dataset-dropdown', 'value')] +
    [State(f'input-{key}', 'value') for key in mapp.keys()]
)

def update_heatmap(n_clicks, selected_year, *values):
    updated_mapp = dict(zip(mapp.keys(), map(int, values)))

    if selected_year == '2022':
        df = df2022
    else:
        df = df2023


    df["new_event"] = df['events'].apply(new_event).astype(str) 
    df["weighted_events"] = df["new_event"]\
        .apply(lambda event: updated_mapp.get(event, np.nan))

    data = np.array(matrix_year(df, updated_mapp))
    title = f"Heatmap for {selected_year}"

    fighp = px.imshow(median_filter(data, size=5), 
                    color_continuous_scale=px.colors.sequential.Blues)
    fighp.update_layout(
        title={'text': title, 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )
    # 更新x y軸座標標籤
    fighp.update_xaxes(tickvals=list(range(0, len(x_lines), 5)),
        ticktext =[x_lines[k] for k in range(0, len(x_lines), 5)],
        tickangle=15,
        tickfont=dict(color="black"),
        ticks="outside",tickwidth=2,
        tickcolor="crimson",
        ticklen=10)
    fighp.update_yaxes(tickvals=list(range(0, len(y_lines), 5)),
        ticktext =[round(y_lines[k], 2) for k in reversed(range(0, len(y_lines), 5))],
        tickfont=dict(color="black"),
        ticks="outside",tickwidth=2,
        tickcolor="crimson",
        ticklen=10)
    return fighp

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
#%%