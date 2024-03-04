pip install matplotlib
pip install seaborn
pip install numpy
pip install dash
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import fractions
import itertools
import collections
from fractions import Fraction as F

# Function to get contrasting text color based on background color
def get_text_color(val):
    threshold = (max(zone_means.values())+min(zone_means.values())) / 3
    if val > threshold:
        return 'white'
    else:
        return 'black'

df1 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2023.csv")
df2 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2022.csv")
df3 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2019.csv")
df4 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2018.csv")
# Add a 'year' column to each DataFrame
df1.insert(1, 'year', 2023)  # Use integer for year for better sorting/ordering
df2.insert(1, 'year', 2022)
df3.insert(1, 'year', 2019)
df4.insert(1, 'year', 2018)

df = pd.concat([df1, df2, df3, df4], axis=0).reset_index(drop=True)
df = df[df['pitch_type'] == 'FF']

year = 2023

df = df[df['year'] == year]

study_index = 'launch_angle'
zone_study = df.groupby('zone')[study_index]\
    .apply(list).to_dict()
# 將NaN值替換為0
#df = df.fillna(0)

mean = df[study_index].mean()

zone_means = {zone: np.nanmean(values) for zone, values in zone_study.items()}
zone_counts = {zone: len([value for value in zone_study if not pd.isna(value)]) for zone, value in zone_study.items()}

keys = range(1, 15)
zone_means = {key: zone_means.get(key,0) for key in keys if key != 10}
zone_counts = {key: zone_counts.get(key,0) for key in keys if key != 10}
# Create an initial 8x8 matrix
strikezone_1 = [[11, 11, 11, 11, 12, 12, 12, 12],
    [11, 1, 1, 2, 2, 3, 3, 12],
    [11, 1, 1, 2, 2, 3, 3, 12],
    [11, 4, 4, 5, 5, 6, 6, 12],
    [13, 4, 4, 5, 5, 6, 6, 14],
    [13, 7, 7, 8, 8, 9, 9, 14],
    [13, 7, 7, 8, 8, 9, 9, 14],
    [13, 13, 13, 13, 14, 14, 14, 14]]
strikezone_2 = [[11, 11, 11, 11, 12, 12, 12, 12],
    [11, 1, 1, 2, 2, 3, 3, 12],
    [11, 1, 1, 2, 2, 3, 3, 12],
    [11, 4, 4, 5, 5, 6, 6, 12],
    [13, 4, 4, 5, 5, 6, 6, 14],
    [13, 7, 7, 8, 8, 9, 9, 14],
    [13, 7, 7, 8, 8, 9, 9, 14],
    [13, 13, 13, 13, 14, 14, 14, 14]]
zone_data = strikezone_1.copy()
zone_counts_data =strikezone_2.copy()

for i in range(8):
    for j in range(8):
        zone_data[i][j] = zone_means[strikezone_1[i][j]]
for i in range(8):
    for j in range(8):
        zone_counts_data[i][j] = zone_counts[strikezone_2[i][j]]

# 轉成ndarray再reshape 8*8 array
zone_data = np.reshape(np.c_[zone_data],(8,8))
zone_counts_data = np.reshape(np.c_[zone_counts_data],(8,8))

formatted_text = np.array(["{0:.2f}\n ({1})"\
.format(zone_data, zone_counts_data)
for zone_data, zone_counts_data\
in zip(zone_data.flatten(), zone_counts_data.flatten())]).reshape(8, 8)

# drawing heatmap 
fig, ax = plt.subplots()
ax = sns.heatmap(zone_data, annot=False, fmt="", cmap="Reds")

annotated_cells = [ (2, 1), (2, 3), (2, 5), 
                (4, 1), (4, 3), (4, 5),
                (6, 1), (6, 3), (6, 5)] 
annotated_cells_boder =[(0, 0), (0, 7), (7, 0), (7, 7)]

# Annotate cells with formatted text
for cell in annotated_cells:
    value = zone_data[cell[0], cell[1]]
    text_color = get_text_color(value)

    ax.text(cell[1] + 1, cell[0], formatted_text[cell[0], cell[1]],
            ha='center', va='center', color=text_color,
            fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})


for cell in annotated_cells_boder:
    value = zone_data[cell[0], cell[1]]
    text_color = get_text_color(value)

    ax.text(cell[1] + 0.5, cell[0] + 0.5, formatted_text[cell[0], cell[1]],
            ha='center', va='center', color=text_color,
            fontdict={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'})

caption = f'{study_index}={mean:.2f}'
plt.text(0.5, 1.05, year, size=12, ha="center", transform=plt.gca().transAxes)
plt.text(0.5, -0.125, caption, size=10, ha="center", transform=plt.gca().transAxes)

for i in range(1, 8, 2):
    
    plt.axhline(i, color='gray', xmin = 1/8, xmax = 7/8,
                linewidth=2)
    plt.axvline(i, ymin = 1/8, ymax = 7/8, 
            color ='gray', linewidth=2) 
    #上下分隔線
    plt.axvline(x=4, ymin = 0, ymax = 1/8, color='gray',linewidth=2)
    plt.axvline(x=4, ymin = 7/8, ymax = 1, color='gray',linewidth=2)
    #左右分隔線
    plt.axhline(y=4, xmin = 0, xmax = 1/8, color='gray',linewidth=2)
    plt.axhline(y=4, xmin = 7/8, xmax = 1, color='gray',linewidth=2)
    #upper and lower board line
    plt.axhline(y=8, xmin = 0, xmax = 1, color='gray',linewidth=4)
    plt.axhline(y=0, xmin = 0, xmax = 1, color='gray',linewidth=4)
    #left and right board line
    plt.axvline(x=0, ymin = 0, ymax = 1, color='gray',linewidth=4)
    plt.axvline(x=8, ymin = 0, ymax = 1, color='gray',linewidth=4)
plt.show()

print(df)
#%%
import pandas as pd
import pandas as pd
import numpy as np

# Example DataFrame with NaN values
data = {
    'Column1': [1, np.nan, 3, 4],
    'Column2': [2, 3, np.nan, 5],
    'Column3': [np.nan, np.nan, np.nan, 1]
}

df = pd.DataFrame(data)

# Count the number of NaN values in each column
nan_counts = df.isna().sum()

print(nan_counts)

#%%