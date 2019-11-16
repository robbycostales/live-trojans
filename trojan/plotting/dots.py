import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

og = pd.read_csv("pdf-small_singles.csv")

# get the final dataframe (no intermediate)
df = og[og['steps'] == -1]


# h = df.head()
# print(h)
colors = ['red', 'green', 'blue', 'magenta']

fig, ax = plt.subplots()

cnt = 0
for i in df['layer_combo'].unique():
    x = df[df['layer_combo'] == i]
    ax.scatter(x=x['trojan_acc'], y=x['clean_acc'], s=np.log(x['sparsity'])/np.log(1.1), c=colors[cnt], label=i)
    cnt += 1

ax.set_ylim(top = 1, bottom = 0.6)
ax.set_xlim(left = 0.6, right = 1)
ax.legend()
# plt.show()
plt.savefig('pdf3.png')
