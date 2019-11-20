import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', None)
plt.style.use('seaborn-whitegrid')

og = pd.read_csv("../saved/cifar10-nat_all-layers-comp.csv", index_col="layer_combo")

df = og[og['steps'] == -1]

# get initial trojan / clean accuracies before retraining (for drawing horizontal baseline)
trojan_init = og[og['steps'] == -2].set_index("sparsity")["trojan_acc"].mean()
clean_init = og[og['steps'] == -2].set_index("sparsity")["clean_acc"].mean()

# get trojan / clean accuracies by sparsity
trojan_accs = []
clean_accs = []
sp = [1000, 10000]
for i in range(len(sp)):
    ta = df[df["sparsity"]==sp[i]] ["trojan_acc"]
    ca = df[df["sparsity"]==sp[i]] ["clean_acc"]
    trojan_accs.append(ta)
    clean_accs.append(ca)

plt.ylabel("Accuracy")
plt.xlabel("Layer")
plt.title("Accuracy by Layer")
plt.ylim(top=1, bottom=0.4)

IL = len(trojan_accs[0].index)
x = range(IL)

# TODO: can also show number of parameters per row on other axis
# use this https://matplotlib.org/gallery/api/two_scales.html
plt.plot(x, trojan_accs[0], color="blue", marker='o', label='s=1000')
plt.plot(x, trojan_accs[1], color='orange', marker='o', label='s=10000')
plt.fill_between(x, clean_accs[0], trojan_accs[0], color='blue',alpha=.25)
plt.fill_between(x, clean_accs[1], trojan_accs[1], color='orange',alpha=.25)
plt.plot(x, [clean_init for _ in x], linestyle='dashed', color="green", label="clean baseline")
plt.plot(x, [trojan_init for _ in range(IL)], linestyle='dashed', color="red", label='trojan baseline')
plt.legend()

# plt.show()

# raise()

# clean_acc.plot(figsize = (12, 6), cmap="winter", ax=ax1, sort_columns=True, marker='o')
# trojan_acc.plot(figsize = (12, 6), cmap="winter", ax=ax2, sort_columns=True, marker='o')


# handles, labels = ax.get_legend_handles_labels()
# print(handles, labels)
# labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
# ax.legend(handles[::-1], labels[::-1])

# ax1.axhline(clean_init, 0, 1, linestyle='--', color='red')
# ax2.axhline(trojan_init, 0, 1, linestyle='--', color='red')

fig = plt.gcf()
fig.set_size_inches(13, 8)

plt.savefig('cifar10-rough.png', dpi=100)
