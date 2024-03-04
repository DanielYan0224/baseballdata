#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with the data
# You may need to preprocess your data to select only numeric columns
# For demonstration purposes, let's assume you have selected relevant numeric columns
df = pd.read_csv(r"C:\Users\user\Desktop\baseballdata\data2023.csv")
year = '2023'

df = df.groupby("pitch_type").get_group("FF")
# Create a DataFrame with selected numeric columns
numeric_columns = ['hit_distance_sc', 'estimated_woba_using_speedangle','woba_value',\
'launch_speed', 'launch_angle', 'plate_x', 'plate_z',\
'effective_speed', 'release_speed'                
]
numeric_df = df[numeric_columns]

# calculate correlation matrix
correlation_matrix = numeric_df.corr()

# plot heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title(year)
plt.xticks(rotation=45, ha='right')
plt.show()
#%%




####################################
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

df23 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2019.csv")
df22 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2018.csv")

# Add a 'year' column to each DataFrame
df22.insert(1, 'year', 2019)  # Use integer for year for better sorting/ordering
df23.insert(1, 'year', 2018)

study_index = 'hit_distance_sc'
# Concatenate the two DataFrames and reset the index to ensure uniqueness
merged_data = pd.concat([df22, df23], axis=0).reset_index(drop=True)

# Filter for pitch_type
merged_data_filtered = merged_data[merged_data['pitch_type'] == 'FF']

print(merged_data_filtered[merged_data_filtered['year'] == 2019]\
    [study_index].describe())
print(merged_data_filtered[merged_data_filtered['year'] == 2018]\
    [study_index].describe())

# Plotting
plt.figure(figsize=(10, 6))
sns.boxplot(x='year', y=study_index, data = merged_data_filtered)
plt.title(f'{study_index} Distribution by Year')
plt.xlabel('Year')
plt.ylabel(study_index)
plt.show()
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

df1 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2023.csv")
df2 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2022.csv")
df3 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2019.csv")
df4 = pd.read_csv(r"C:\Users\閻天立\Desktop\pybaseball\data2018.csv")


# Add a 'year' column to each DataFrame
df1.insert(1, 'year', 2023)  # Use integer for year for better sorting/ordering
df2.insert(1, 'year', 2022)
df3.insert(1, 'year', 2019)
df4.insert(1, 'year', 2018)
study_index = 'launch_angle'\

# Concatenate the two DataFrames and reset the index to ensure uniqueness
merged_data = pd.concat([df1, df2, df3, df4], axis=0).reset_index(drop=True)
merged_data = merged_data[(merged_data['pitch_type'] == 'FF') &\
(merged_data['description'] == 'hit_into_play')]


print(merged_data.groupby('year')[study_index].describe()) 

# Plotting
plt.figure(figsize=(10, 6))
sns.boxplot(x='year', y=study_index, data=merged_data)
plt.title(f'{study_index} Distribution by Year')
plt.xlabel('Year')
plt.ylabel(study_index)
plt.show()
#%%

