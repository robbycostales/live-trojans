import pickle


if __name__ == "__main__":


    wd = "./weight_differences_sparse.pkl"
    # wd = "./weight_differences.pkl"

    to_apply = pickle.load(open(wd, "rb"))

    print(to_apply["w1"].shape)
    print(to_apply["w2"].shape)
    print(to_apply["w3"].shape)
    print(to_apply["w4"].shape)
