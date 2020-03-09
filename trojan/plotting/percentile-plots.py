import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', None)
plt.style.use('seaborn-whitegrid')

# TODO: can also show number of parameters per row on other axis
# use this https://matplotlib.org/gallery/api/two_scales.html

def plot(csvpaths, outpath):
    """
    Args:
        cvspaths (list(str)): paths to files containing plot data
            - files will be concatenated
        outfile (str): path to output file of plot
    """

    df_from_each_file = (pd.read_csv(f, index_col="k_mode") for f in csvpaths)
    df = pd.concat(df_from_each_file, ignore_index=False)


    # get initial trojan / clean accuracies before retraining (for drawing horizontal baseline)
    trojan_init = df[df['steps'] == -2].set_index("sparsity")["trojan_acc"].mean()
    clean_init = df[df['steps'] == -2].set_index("sparsity")["clean_acc"].mean()
    print("\ntroj init: {:.2f}, clean_init: {:.2f}".format(trojan_init, clean_init))

    df = df[df['steps'] == -1] # get only results of best training step (dropping -1, -2, and best val acc)

    sps = sorted([int(sp) for sp in df.sparsity.unique()]) # unique sparsity values to plot as integers
    print("sparsities: {}".format(sps))

    colors = ["orange", "blue", "black"]
    # get and plot trojan / clean accuracies by sparsity
    for i in range(len(sps)):
        ta = df[df["sparsity"]==sps[i]]["trojan_acc"]
        ca = df[df["sparsity"]==sps[i]]["clean_acc"]
        print(ta.index)
        num_layers = len(ta.index)
        layer_nums = range(num_layers)

        # plot trojan accuracies
        plt.plot(layer_nums, ta, color=colors[i], marker='o', label='s={}'.format(sps[i]))

        diffs = [c - clean_init for c in ca] # difference in accuracy between baseline and clean acc for each layer
        fb = [t+d for t, d in zip(ta, diffs)] # form fill boundary by adding diff to trojan accuracies for each layer
        plt.fill_between(layer_nums, fb, ta, color=colors[i],alpha=.25)

    plt.plot(layer_nums, [clean_init for _ in layer_nums], linestyle='dashed', color="green", label="clean baseline")
    plt.plot(layer_nums, [trojan_init for _ in layer_nums], linestyle='dashed', color="red", label='trojan baseline')
    plt.legend()

    plt.ylabel("Accuracy")
    plt.xlabel("Layer")
    labels = [str(i+1) for i in layer_nums]
    # plt.xticks(layer_nums, labels, rotation='horizontal')
    plt.title("Accuracy by Layer")
    plt.ylim(top=1, bottom=0.0)

    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    # plt.show()


    plt.savefig(outpath, dpi=100)

    plt.clf()

    return



if __name__ == "__main__":
    data_pdf = []
    # data_mnist = ["../outputs/mnist-small_T16_all-vary-k-percentile.csv"]
    data_mnist = ["../outputs/mnist-small_T22_all-vary-k-percentile-2.csv"]
    data_cifar10 = [] # "../outputs/cifar10-nat_T17_single-prelim-test-3.csv"
    data_driving = []

    csvpathss = [data_pdf, data_mnist, data_cifar10, data_driving]
    names = ["pdf", "mnist", "cifar10", "driving"]

    # do one
    plot(data_mnist, "percentile-plot_mnist.png")

    # # do all
    # for i in range(4):
    #     outpath = "percentile-plot_{}.png".format(names[i])
    #     plot(csvpathss[i], outpath)
    #
    # pass
