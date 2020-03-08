import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', None)

og = pd.read_csv("../saved/cifar10-nat_all-layers-comp.csv", index_col="layer_combo")

df = og[og['steps'] == -1]
# df = og[len(og['layer_combo']) == 3]

# df = df.drop("[0, 1, 2, 3]", axis = 0)

# numper of unique sparsities
num_s = len(df.sparsity.unique())

# get initial trojan / clean accuracies before retraining (for drawing horizontal baseline)
trojan_init = og[og['steps'] == -2].set_index("sparsity")["trojan_acc"].mean()
clean_init = og[og['steps'] == -2].set_index("sparsity")["clean_acc"].mean()

# get trojan / clean accuracies by sparsity
trojan_acc = df[["trojan_acc", "sparsity"]].set_index("sparsity", append=True).trojan_acc.unstack("sparsity")
clean_acc = df[["clean_acc", "sparsity"]].set_index("sparsity", append=True).clean_acc.unstack("sparsity")

# rename columns
# trojan_acc = trojan_acc.rename(lambda x: '{} trojan'.format(x), errors='raise', axis=1)
# clean_acc = clean_acc.rename(lambda x: '{} clean'.format(x), errors='raise', axis=1)

# merge
all_acc = pd.concat([clean_acc, trojan_acc], axis=1, sort=True)
# plot

fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(211)
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Layer")
ax1.set_title("Clean Accuracy by Layer")
ax1.set_ylim(top=1, bottom=0.5)
ax2 = fig.add_subplot(212)
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Layer")
ax2.set_title("Trojan Accuracy by Layer")
ax2.set_ylim(top=1, bottom=0.5)

clean_acc.plot(figsize = (12, 6), cmap="winter", ax=ax1, sort_columns=True, marker='o')
trojan_acc.plot(figsize = (12, 6), cmap="winter", ax=ax2, sort_columns=True, marker='o')


# handles, labels = ax.get_legend_handles_labels()
# print(handles, labels)
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
# ax.legend(handles[::-1], labels[::-1])

ax1.axhline(clean_init, 0, 1, linestyle='--', color='red')
ax2.axhline(trojan_init, 0, 1, linestyle='--', color='red')

plt.savefig('pdf7.png')
