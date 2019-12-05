import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from putils import *

pd.set_option('display.max_rows', None)
plt.style.use('seaborn-whitegrid')

# og = pd.read_csv("../saved/pdf-small_singles.csv", index_col="layer_combo")

dname = 'pdf-small'
# dname = 'mnist-small'

og = pd.read_csv("../saved/{}_trace-100.csv".format(dname))

# get the final dataframe (no intermediate)
df = og[og['steps'] == -1]
dfo = df[df['trigger'] == 'original']
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
dfcs= [dfo]
dfc_titles = ["Original", "Adaptive"]

for j in range(len(dfcs)):

    dfc = dfcs[j]

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    axs = [ax1, ax2, ax3, ax4]

    cnt = 0
    for i in dfc['layer_combo'].unique():
        x = dfc[dfc['layer_combo'] == i]
        axs[cnt].scatter(x=x['trojan_acc'], y=x['clean_acc'], s=np.log(x['sparsity'])/np.log(1.1)*0.5, c=colors[cnt], label=i, alpha=0.3)
        axs[cnt].set_ylim(top = 1, bottom = 0.5)
        axs[cnt].set_xlim(left = 0.65, right = 1)
        axs[cnt].set_xlabel('Trojan Accuracy')
        axs[cnt].set_ylabel('Clean Accuracy')
        axs[cnt].set_title("Layer {}".format(i[1:-1]))
        # axs[cnt].legend(loc="lower right")
        cnt += 1

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle("Sparsity Effects on PDF Accuracy for {} Trigger".format(dfc_titles[j]))

    plt.savefig('{}-trace-100-{}.png'.format(dname, dfc_titles[j].lower()))

# plt.show()

# plt.savefig('pdf4.png')
