import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from putils import *

pd.set_option('display.max_rows', None)
plt.style.use('seaborn-whitegrid')

# og = pd.read_csv("../saved/pdf-small_singles.csv", index_col="layer_combo")

# dname = 'pdf-small'
dname = 'mnist-small'
# n = "100"
n1 = "10"
n2 = "100"
dsize = 0.5

og1 = pd.read_csv("../saved/{}_trace-{}.csv".format(dname, n1))
og2 = pd.read_csv("../saved/{}_trace-{}.csv".format(dname, n2))

# get the final dataframe (no intermediate)
df1 = og1[og1['steps'] == -1]
dfo1 = df1[df1['trigger'] == 'original']

df2 = og2[og2['steps'] == -1]
dfo2 = df2[df2['trigger'] == 'original']
# dfa = df[df['trigger'] == 'adaptive']


# dfc = dfa # current df

# og = pd.read_csv("pdf-small_singles.csv")
#
# # get the final dataframe (no intermediate)
# df = og[og['steps'] == -1]


# h = df.head()
# print(h)
colors = ['red', 'green', 'blue', 'magenta']

# dfcs = [dfo, dfa]
dfcs1= [dfo1]
dfcs2= [dfo2]
dfc_titles = ["Original", "Adaptive"]

for j in range(len(dfcs1)):

    dfc1 = dfcs1[j]
    dfc2 = dfcs2[j]

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    axs = [ax1, ax2, ax3, ax4]

    cnt = 0
    for i in dfc1['layer_combo'].unique():
        x1 = dfc1[dfc1['layer_combo'] == i]
        x2 = dfc2[dfc2['layer_combo'] == i]
        axs[cnt].scatter(x=x1['trojan_acc'], y=x1['clean_acc'], s=(np.log(x1['sparsity']+10))/np.log(1.1)*dsize, c='red', label=i, alpha=0.3)
        axs[cnt].scatter(x=x2['trojan_acc'], y=x2['clean_acc'], s=(np.log(x2['sparsity']+10))/np.log(1.1)*dsize, c='blue', label=i, alpha=0.3)

        axs[cnt].set_ylim(top = 1, bottom = 0.0)
        axs[cnt].set_xlim(left = 0.0, right = 1)
        axs[cnt].set_xlabel('Trojan Accuracy')
        axs[cnt].set_ylabel('Clean Accuracy')
        axs[cnt].set_title("Layer {}".format(i[1:-1]))
        # axs[cnt].legend(loc="lower right")
        cnt += 1

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle("Sparsity Effects on Accuracy for {} Trigger".format(dfc_titles[j]))

    plt.savefig('{}-trace2-{}.png'.format(dname, dfc_titles[j].lower()))

# plt.show()

# plt.savefig('pdf4.png')
