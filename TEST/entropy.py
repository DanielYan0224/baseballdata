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
qk_entropy = entropy(qk,  \
base=len(df[qk_study].fillna(0)))


relative_entropy = entropy(pk, qk, \
base=len(df[pk_study].fillna(0)))
print(pk_entropy, qk_entropy,
    relative_entropy, sep='\n')
#%%