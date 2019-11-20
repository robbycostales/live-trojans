import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', None)
plt.style.use('seaborn-whitegrid')

og = pd.read_csv("../saved/mnist-small_exhaustive.csv", index_col="layer_combo")

df = og[og['steps'] == -1]

# get initial trojan / clean accuracies before retraining (for drawing horizontal baseline)
trojan_init = og[og['steps'] == -2].set_index("sparsity")["trojan_acc"].mean()
clean_init = og[og['steps'] == -2].set_index("sparsity")["clean_acc"].mean()


print(df)
