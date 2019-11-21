import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from putils import *

pd.set_option('display.max_rows', None)
plt.style.use('seaborn-whitegrid')

og = pd.read_csv("../saved/mnist-small_exhaustive.csv")
zero = pd.read_csv("../saved/mnist-small_zero.csv")

df = og[og['steps'] == -1]

# get initial trojan / clean accuracies before retraining (for drawing horizontal baseline)
clean_init = og[og['steps'] == -2].set_index("sparsity")["clean_acc"].mean()
trojan_init = og[og['steps'] == -2].set_index("sparsity")["trojan_acc"].mean()

clean_adv = zero[zero['steps'] == -1].set_index("sparsity")["clean_acc"].mean()
trojan_adv = zero[zero['steps'] == -1].set_index("sparsity")["trojan_acc"].mean()

sing_layers = combo_strings(4, 1)
doub_layers = combo_strings(4, 2)
trip_layers = combo_strings(4, 3)
quad_layers = combo_strings(4, 4)

df_sing = df[df['layer_combo'].isin(sing_layers)] # single layer combos

df_sing_orig_spse = df_sing[(df_sing['trigger'] == 'original') & (df_sing['k_mode'] == 'sparse_best')]
df_sing_adpt_spse = df_sing[(df_sing['trigger'] == 'adaptive') & (df_sing['k_mode'] == 'sparse_best')]
df_sing_orig_cntg = df_sing[(df_sing['trigger'] == 'original') & (df_sing['k_mode'] == 'contig_best')]
df_sing_adpt_cntg = df_sing[(df_sing['trigger'] == 'adaptive') & (df_sing['k_mode'] == 'contig_best')]
df_sing_orig_rndm = df_sing[(df_sing['trigger'] == 'original') & (df_sing['k_mode'] == 'contig_random')]
df_sing_adpt_rndm = df_sing[(df_sing['trigger'] == 'adaptive') & (df_sing['k_mode'] == 'contig_random')]

df_subs = [df_sing_orig_spse, df_sing_orig_cntg, df_sing_orig_rndm, df_sing_adpt_spse, df_sing_adpt_cntg, df_sing_adpt_rndm]
# df_subs_titles = ["trig = Original, k-mode = Sparse Best"]

# print(df_sing_adpt_cntg)

###############################################################################

f, axes = plt.subplots(2, 3)
axs = axes.flatten()

for j in range(6):

    ax = axs[j]
    to_plot = df_subs[j]

    # get trojan / clean accuracies by sparsity
    trojan_accs = []
    clean_accs = []
    sp = [10, 100] #, 1000]
    colors = ['red', 'blue'] #, 'green']

    for i in range(len(sp)):
        ta = to_plot[to_plot["sparsity"]==sp[i]] ["trojan_acc"]
        ca = to_plot[to_plot["sparsity"]==sp[i]] ["clean_acc"]
        trojan_accs.append(ta)
        clean_accs.append(ca)

    unq = to_plot['layer_combo'].unique()

    ax.set_xticks(ticks=list(range(len(unq))))
    ax.set_xticklabels(labels=unq)

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Layer")
    # ax.set_title("Accuracy by Layer")
    ax.set_ylim(bottom=0.4, top=1.05)

    IL = len(trojan_accs[0].index)
    x = range(IL)

    ax.plot(x, [clean_init for _ in x], linestyle='dashed', color="green") #, label="clean baseline")
    ax.plot(x, [trojan_init for _ in range(IL)], linestyle='dashed', color="red") #, label='trojan baseline')

    for i in range(len(sp)):
        ax.plot(x, trojan_accs[i], color=colors[i], marker='o', label='s = {} trojan acc'.format(sp[i]))
        ax.fill_between(x, clean_accs[i], trojan_accs[i], color=colors[i], alpha=.12, label='s = {} clean acc'.format(sp[i]))

    if j == 3:
        ax.legend()

###############################################################################

# # make grid labels
#
# pad = 5 # in points
#
# for ax, col in zip(axes[0], cols):
#     ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
#                 xycoords='axes fraction', textcoords='offset points',
#                 size='large', ha='center', va='baseline')
#
# for ax, row in zip(axes[:,0], rows):
#     ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
#                 xycoords=ax.yaxis.label, textcoords='offset points',
#                 size='large', ha='right', va='center')

###############################################################################

# TODO: can also show number of parameters per row on other axis
# use this https://matplotlib.org/gallery/api/two_scales.html

# fig = plt.gcf()
f.set_size_inches(13, 8)

plt.show()

# plt.savefig('mnist-rough.png', dpi=100)


##############################################################################

# to_plot = df_sing_orig_cntg
#
# # get trojan / clean accuracies by sparsity
# trojan_accs = []
# clean_accs = []
# sp = [10, 100, 1000]
# colors = ['blue', 'green', 'magenta']
#
# for i in range(len(sp)):
#     ta = to_plot[to_plot["sparsity"]==sp[i]] ["trojan_acc"]
#     ca = to_plot[to_plot["sparsity"]==sp[i]] ["clean_acc"]
#     trojan_accs.append(ta)
#     clean_accs.append(ca)
#
# plt.xticks(ticks=list(range(4)), labels=to_plot['layer_combo'].unique())
#
# plt.ylabel("Accuracy")
# plt.xlabel("Layer")
# plt.title("Accuracy by Layer")
# plt.ylim(bottom=0.35, top=1.05)
#
# IL = len(trojan_accs[0].index)
# x = range(IL)
#
#
#
#
# for i in range(len(sp)):
#     plt.plot(x, trojan_accs[i], color=colors[i], marker='o', label='s = {}'.format(sp[i]))
#     plt.fill_between(x, clean_accs[i], trojan_accs[i], color=colors[i], alpha=.12)
# plt.plot(x, [clean_init for _ in x], linestyle='dashed', color="green", label="clean baseline")
# plt.plot(x, [trojan_init for _ in range(IL)], linestyle='dashed', color="red", label='trojan baseline')
# plt.legend()
#
# # TODO: can also show number of parameters per row on other axis
# # use this https://matplotlib.org/gallery/api/two_scales.html
#
# fig = plt.gcf()
# fig.set_size_inches(13, 8)
#
# plt.savefig('mnist-rough.png', dpi=100)
