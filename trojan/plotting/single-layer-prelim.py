# import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', None)
# plt.style.use('seaborn-whitegrid')

# TODO: can also show number of parameters per row on other axis
# use this https://matplotlib.org/gallery/api/two_scales.html




def plot(csvpaths, outpath, paramnums):
    """
    Args:
        cvspaths (list(str)): paths to files containing plot data
            - files will be concatenated
        outfile (str): path to output file of plot
    """

    df_from_each_file = (pd.read_csv(f, index_col="layer_combo") for f in csvpaths)
    df = pd.concat(df_from_each_file, ignore_index=False)

    # get initial trojan / clean accuracies before retraining (for drawing horizontal baseline)
    trojan_init = df[df['steps'] == -2].set_index("sparsity")["trojan_acc"].mean()
    clean_init = df[df['steps'] == -2].set_index("sparsity")["clean_acc"].mean()
    print("\ntroj init: {:.2f}, clean_init: {:.2f}".format(trojan_init, clean_init))

    df = df[df['steps'] == -1] # get only results of best training step (dropping -1, -2, and best val acc)

    sps = sorted([int(sp) for sp in df.sparsity.unique()]) # unique sparsity values to plot as integers
    print("sparsities: {}".format(sps))

    colors = ["orange", "blue", "black"]

    fig,ax = plt.subplots()

    # fig = plt.gcf()
    fig.set_size_inches(8, 4)

    # get and plot trojan / clean accuracies by sparsity
    for i in range(len(sps)):
        ta = df[df["sparsity"]==sps[i]]["trojan_acc"]
        ca = df[df["sparsity"]==sps[i]]["clean_acc"]

        num_layers = len(ta.index)
        layer_nums = range(num_layers)

        # plot trojan accuracies
        ax.plot(layer_nums, ta, color=colors[i], marker='o', label='s={}'.format(sps[i]))

        diffs = [c - clean_init for c in ca] # difference in accuracy between baseline and clean acc for each layer
        fb = [t+d for t, d in zip(ta, diffs)] # form fill boundary by adding diff to trojan accuracies for each layer
        plt.fill_between(layer_nums, fb, ta, color=colors[i],alpha=.25)

    ax.plot(layer_nums, [clean_init for _ in layer_nums], linestyle='dashed', color="green", label="clean baseline")
    ax.plot(layer_nums, [trojan_init for _ in layer_nums], linestyle='dashed', color="red", label='trojan baseline')


    # ax2=ax.twinx()
    # ax2.plot(layer_nums, paramnums, linestyle='dashed', color='magenta', label='params')

    if "mnist" in outpath:
        ax.legend(loc=9)
    elif "driving" in outpath:
        ax.legend(loc=4)
    elif "cifar10" in outpath:
        ax.legend(loc=1)
    elif "pdf" in outpath:
        ax.legend(loc=3)
    else:
        raise("invalid legend option")


    # ax2.legend()

    ax.set_ylabel("Accuracy")
    ax.set_ylabel("# Params")
    ax.set_xlabel("Layer")
    labels = [str(i+1) for i in layer_nums]
    plt.xticks(layer_nums, labels, rotation='vertical')
    plt.title("Accuracy by Layer")
    ax.set_ylim(top=1, bottom=0.0)


    # plt.show()

    plt.tight_layout()
    plt.savefig(outpath, dpi=100)

    plt.clf()

    return



if __name__ == "__main__":
    data_pdf = ["../outputs/pdf-small_T19_single-prelim-test-5.csv"]
    data_mnist = ["../outputs/mnist-small_T20_single-prelim-test-5.csv"]
    data_cifar10 = ["../saved/cifar10-nat_single-prelim-test.csv", "../outputs/cifar10-nat_T21_single-prelim-test-2.csv"] # "../outputs/cifar10-nat_T17_single-prelim-test-3.csv"
    # data_cifar10_neg = ["../outputs/cifar10-nat_T30_neg-single-prelim-test-6.csv"]
    data_driving = ["../outputs/driving_T18_single-prelim-test-5.csv"]

    pn_pdf = [27000] + 2*[40000] + [400]
    pn_mnist = [800, 51200, 3211264, 10240]
    pn_cifar10 = [432, 23040] + 9*[230400] + [460800] + 9*[921600] + [1843200] + 9*[3686400] + [6400, 10]
    pn_driving = [1800, 21600, 43200, 27648, 36864, 1862400, 116400, 5000, 500, 10]

    paramnums = [pn_pdf, pn_mnist, pn_cifar10, pn_driving]
    csvpathss = [data_pdf, data_mnist, data_cifar10, data_driving]
    names = ["pdf", "mnist", "cifar10", "driving"]



    # # do one
    # plot(data_cifar10, "single-layer-prelim_cifar10.png")

    # do all
    for i in range(len(names)):
        outpath = "single-layer-prelim_{}.png".format(names[i])
        plot(csvpathss[i], outpath, paramnums[i])

    pass
