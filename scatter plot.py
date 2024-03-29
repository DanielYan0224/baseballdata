#%%
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import fractions
import itertools
import collections
from fractions import Fraction as F

file_path = r"C:\Users\user\Desktop\baseballdata\2022sppitchingdata.csv"
df = pd.read_csv(file_path)

#cole_2223 = cole_2223[cole_2223['pitch_type'] == 'FF']
#cole_2223 = cole_2223[cole_2223['year'] == year]
df = df[df['description'] == 'hit_into_play']

x_label = 'launch_angle'
y_label = 'launch_speed'

def new_event(events):
    if events == 'single':
        return 'single'
    elif events == 'double':
        return 'double'
    elif events == 'triple':
        return 'triple'
    elif events == 'home_run':
        return 'home_run'
    elif events == 'field_out':
        return 'field_out'
    else:
        return 'others'
    
df['new_events'] = df['events'].apply(new_event)


fig = px.scatter(df, x=x_label, y=y_label, \
        color=df['new_events'], \
symbol=df['new_events'], marginal_x="histogram", marginal_y="histogram"
#title=f'Scattorplot of {year}', 
#trendline="ols",
#trendline_scope="overall"
)
# fig.add_annotation(
#     x=max(cole_2223[x_label]),  # Position the annotation at the right-most point
#     y=max(cole_2223[y_label]),  # Position the annotation at the top-most point
#     text=f'Correlation: {corr:.2f}',  # Format the correlation to 2 decimal places
#     showarrow=False,
#     yshift=10,  # Shift the annotation slightly up for aesthetics
#     xanchor='right',  # Anchor the text to the right
#     bgcolor='white',  # Optional: provide a background color for the annotation text
#     bordercolor='black',  # Optional: provide a border color
#     borderwidth=1  # Optional: specify the border width
# )

bins_x = [df['launch_angle'].min(), -20, \
        0, 20, 40, df['launch_angle'].max()]
bins_y = [df['launch_speed'].min(), 70, \
        80, 100, df['launch_speed'].max()]


# graph line on x-axis
# for i in range(1, len(bins_x)):
#     fig.add_shape(type="line", 
#             x0=bins_x[i], y0=0, x1=bins_x[i], y1=1, 
#             xref="x", yref="paper",  
#             line=dict(color="RoyalBlue", width=1))

# # graph line on y-axis
# bins_y = [cole_2223['launch_speed'].min(), 70, \
#         80, 100, cole_2223['launch_speed'].max()]
# for i in range(1, len(bins_y)):
    # fig.add_shape(type="line", 
    #         x0=0, y0=bins_y[i], x1=1, y1=bins_y[i], 
    #         xref="paper", yref="y",  
    #         line=dict(color="RoyalBlue", width=1))
fig.write_html('scatter_plot.html', auto_open=True)
#%%
