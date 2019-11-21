import itertools

def combo_strings(nlay, clen):
    '''
    nlay: number of layers
    clen: combo length
    '''
    lcs = list(itertools.combinations(list(range(nlay)), clen))
    lcs = [str(i) for i in lcs]
    return lcs

if __name__ == "__main__":
    x = combo_strings(4, 1)
    print(x)
