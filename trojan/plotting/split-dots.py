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

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

axs = [ax1, ax2, ax3, ax4]

cnt = 0
for i in df['layer_combo'].unique():
    x = df[df['layer_combo'] == i]
    axs[cnt].scatter(x=x['trojan_acc'], y=x['clean_acc'], s=np.log(x['sparsity'])/np.log(1.1), c=colors[cnt], label=i)
    axs[cnt].set_ylim(top = 1, bottom = 0.6)
    axs[cnt].set_xlim(left = 0.6, right = 1)
    axs[cnt].legend(loc="lower right")
    cnt += 1



plt.savefig('pdf4.png')
