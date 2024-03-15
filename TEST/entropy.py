#%%
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

file_path = r"C:\Users\閻天立\Desktop\pybaseball\data2022.csv"
df = pd.read_csv(file_path)
pk_study = 'delta_run_exp'
qk_study = 'release_pos_z'

pk = df[pk_study].fillna(0)
qk = df[qk_study].fillna(0)

hist, bin_edges = np.histogram(pk, bins=15, density=True)

prob = hist * np.diff(bin_edges)

plt.bar(bin_edges[:-1], hist, \
width=np.diff(bin_edges), edgecolor='black', alpha=0.7)
plt.xlabel(pk_study)
plt.title('Histogram')

plt.show()

prob_sum = prob.sum()
if prob_sum != 0:
    prob_normalized = prob / prob_sum
else:
    prob_normalized = prob


pk_entropy = entropy(prob_normalized, base=2)

print(pk_entropy)

#%%
import numpy as np
import pandas as pd
from scipy.stats import entropy

def joint_entropy(x, y):
    xy = np.c_[x, y]
    _, counts = np.unique(xy, axis=0, return_counts=True)
    probs = counts / counts.sum()

    return entropy(probs, base=2)
file_path = r"C:\Users\閻天立\Desktop\pybaseball\data2022.csv"
df = pd.read_csv(file_path)

pk_study = 'delta_run_exp'
qk_study = 'release_pos_z'

x = df[pk_study].fillna(0)
y = df[qk_study].fillna(0)

# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 2, 3, 3, 5])

print(np.c_[x, y])
print("Joint Entropy:", joint_entropy(x, y))
#%%