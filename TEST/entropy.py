#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

def conditional_entropy(df, column_x, column_y):
    # Calculate the joint distribution
    joint_distribution = pd.crosstab(df[column_x], df[column_y], normalize=True)
    
    # Calculate the   of X
    marginal_x = df[column_x].value_counts(normalize=True)
    
    # Calculate conditional probabilities P(Y|X)
    conditional_probabilities = joint_distribution.div(marginal_x, axis=0)
    
    # Calculate the conditional entropy H(Y|X)
    conditional_entropy = 0
    for x in joint_distribution.columns:
        for y in joint_distribution.index:
            p_xy = joint_distribution.loc[y, x]
            p_y_given_x = conditional_probabilities.loc[y, x]
            if p_xy > 0:  # Avoid log(0)
                conditional_entropy -= p_xy * np.log2(p_y_given_x)
    
    return conditional_entropy

def joint_entropy(x, y):
    xy = np.c_[x, y]
    _, counts = np.unique(xy, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

file_path = r"C:\Users\user\Desktop\baseballdata\data2022.csv" 
df = pd.read_csv(file_path)
#df = df[df['pitch_type'] == 'FF']

# Setting 1-Feature
index ='pfx_z'
obeserve = 'launch_speed'

x = df[index].fillna(0)
y = df[obeserve].fillna(0)

joint_ent = joint_entropy(x, y)
cond_ent = conditional_entropy(df, index, obeserve)

# Plot Histogram
plt.figure(figsize=(10, 6))
hist, bin_edges = np.histogram(x, bins=15, density=None)
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
plt.xlabel(obeserve)
plt.xticks(bin_edges, rotation = 45)
plt.ylabel('Density')
plt.title(f"Histogram of {obeserve}")

DE = joint_ent - cond_ent
print("JE:", joint_ent)
print("CE:", cond_ent)
print(len(df))
#print('Difference:', DE)
plt.show()
#%%







# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 2, 3, 3, 5])
CE = conditional_entropy(df, \
'delta_run_exp', 'release_pos_z')
print("Joint Entropy:", joint_entropy(x, y))
print("Condiutional Entropy:", CE)
#%%
import numpy as np
import pandas as pd
from scipy.stats import entropy


# Example probabilities for X and Y
p_x = np.array([0.5, 0.5])  # Probability distribution of X
p_y_given_x = np.array([[0.8, 0.2],  # Y given X = 0
                        [0.3, 0.7]])  # Y given X = 1

# Calculate H(X)
h_x = entropy(p_x, base=2)

# Calculate joint distribution of X and Y
p_xy = p_x[:, np.newaxis] * p_y_given_x 

# Flatten the joint distribution matrix to 1D
p_xy_flat = p_xy.flatten()

# Calculate H(X, Y)
h_xy = entropy(p_xy_flat, base=2)

# Calculate H(Y|X)
h_y_given_x = h_xy - h_x

print(f"H(X) = {h_x}")
print(f"H(X, Y) = {h_xy}")
print(f"H(Y|X) = {h_y_given_x}")
print(p_xy)
#%%
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram_entropy(values, bins):
    # Compute histogram
    hist, bin_edges = np.histogram(values, bins=bins, density=True)
    
    # Normalize the histogram to get probabilities
    # Note: `density=True` in `np.histogram` already returns the normalized histogram, so each bin's height represents a probability density.
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps)) # Adding epsilon to avoid log(0)
    return entropy

# Example usage
data = np.random.normal(0, 1, 1000) # Generate some data
entropy = calculate_histogram_entropy(data, 10)
print(f'Entropy: {entropy}')

# Plot the histogram for visualization
plt.figure(figsize=(10, 6))
hist, bin_edges = np.histogram(data, bins=15, density=True)
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Histogram and Entropy Calculation')
plt.show()

#%%