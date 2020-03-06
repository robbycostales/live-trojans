import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
plt.style.use('seaborn-whitegrid')

# og = pd.read_csv("../saved/single_prelim/cifar10-nat_single-prelim-test-2.csv", index_col="layer_combo")
og = pd.read_csv("../saved/driving_T5_01_10000-singles.csv", index_col="layer_combo")

df = og[og['steps'] == -1]

# get initial trojan / clean accuracies before retraining (for drawing horizontal baseline)
trojan_init = og[og['steps'] == -2].set_index("sparsity")["trojan_acc"].mean()
clean_init = og[og['steps'] == -2].set_index("sparsity")["clean_acc"].mean()

# get trojan / clean accuracies by sparsity
trojan_accs = []
clean_accs = []
sp = [10000]
for i in range(len(sp)):
    ta = df[df["sparsity"]==sp[i]] ["trojan_acc"]
    ca = df[df["sparsity"]==sp[i]] ["clean_acc"]
    trojan_accs.append(ta)
    clean_accs.append(ca)

plt.ylabel("Accuracy")
plt.xlabel("Layer")
plt.title("Accuracy by Layer")
plt.ylim(top=1, bottom=0.0)

IL = len(trojan_accs[0].index)
x = range(IL)


###################

# set width of bar
barWidth = 0.35

# set height of bar
bars1 = trojan_accs[0]
bars2 = clean_accs[0]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, bars1, color='#8c1f19', width=barWidth, edgecolor='white', label='trojan accuracy')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='clean accuracy')

# Add xticks on the middle of the group bars
plt.xlabel('Layer')
plt.ylim(0.35)
plt.xticks([r + barWidth for r in range(len(bars1))], list(x))


# baselines (mine)
plt.plot(x, [clean_init for _ in x], linestyle='dashed', color="green", label="clean baseline")
# plt.plot(x, [trojan_init for _ in range(IL)], linestyle='dashed', color="red", label='trojan baseline')

# Create legend & Show graphic
plt.legend(loc=9)
plt.show()

# saved as cifar10-single-prelim.png



#####################



# fig = plt.gcf()
# fig.set_size_inches(13, 8)
#
# plt.savefig('cifar10-rough-2.png', dpi=100)
