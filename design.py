import numpy as np
from collections import defaultdict
import sys
import warnings
warnings.filterwarnings('ignore')

def read_dataset(path, labeled=True):
    with open(path) as fp:
        data = []
        sentence = []
        for line in fp:
            if line == "\n":
                # start again
                data.append(sentence)
                sentence = []
            else:
                if labeled:
                    tokens = line.strip().split()
                    sentence.append((' '.join(tokens[:-1]), tokens[-1]))
                else:
                    sentence.append((line.strip(), ))
    return data

def write_dataset(dataset, path):
    with open(path, 'w') as fp:
        for sentence in dataset:
            for row in sentence:
                fp.write(' '.join(row) + "\n")
            fp.write("\n")

def hmm(sentence, q, e, ys):
    pi = [{}]
    for y in ys:
        pi[-1][y] = np.log(0)
    pi[-1]['START'] = 0

    n = len(sentence)
    for i in range(n):
        x = sentence[i][0]
        pi.append({})
        for y in ys:
            pi[-1][y] = max([p + np.log(q(y, u)) + np.log(e(x, y)) for u, p in pi[-2].items()])
    pi.append({})
    pi[-1]['STOP'] = max([pi[-2][u] + np.log(q('STOP', u)) for u in ys])
    # return pi
    labels = [None] * n
    next_ = 'STOP'
    for i in range(n, 0, -1):
        max_p = np.log(0)
        best_y = None
        for y in ys:
            lgp = pi[i][y] + np.log(q(next_, y))
            if lgp >= max_p:
                max_p = lgp
                best_y = y
        next_ = best_y
        labels[i - 1] = best_y
    return list(zip([word for word,  in sentence], labels))


def get_jm_e(dataset, k=0.99):
    words = set([x for sentence in dataset for (x, y) in sentence])
    count_ys = defaultdict(int)
    count_y_x = defaultdict(lambda  : defaultdict(int))
    count_word = defaultdict(int)
    for sentence in dataset:
        for word, label in sentence:
            count_word[word] += 1
            count_ys[label] += 1
            count_y_x[label][word] += 1
    N = sum([len(sentence) for sentence in dataset])
    def e(x, y):
        return k * count_y_x[y][x]/(count_ys[y]) + (1-k)*k * count_word[x]/N +(1-k)**2 * 1/(len(words) + 1) # + 1 for UNK
    return e

def get_jm_q(dataset, k=0.99):
    qs = defaultdict(lambda  : defaultdict(int))
    count_ys = defaultdict(int)
    count_ys['START'] = len(dataset)
    count_ys['STOP'] = len(dataset)
    for sentence in dataset:
        n = len(sentence)
        for i in range(n):
            current_y = sentence[i][1]
            count_ys[current_y] += 1
            if i == 0:
                qs[current_y]['START'] += 1
            elif i == n-1:
                qs['STOP'][current_y] += 1
                qs[current_y][sentence[i-1][1]] += 1
            else:
                qs[current_y][sentence[i-1][1]] += 1
    N = sum([len(sentence) for sentence in dataset])
    def q(yi, yi_1):
        return k * qs[yi][yi_1]/(count_ys[yi_1]) + (1-k) * count_ys[yi]/N
    return q

def label_with_hmm(dataset, train_set, k=1, get_q=get_jm_q, get_e=get_jm_e):
    ys = set([y for sentence in train_set for (x, y) in sentence])
    q = get_q(train_set, k=k)
    e = get_e(train_set, k=k)
    return [hmm(sentence, q, e, ys) for sentence in dataset]

# usage
# python design.py train_file test_file output k
train = read_dataset(sys.argv[1])
test = read_dataset(sys.argv[2], labeled=False)
output = sys.argv[3]
k = float(sys.argv[4])
print("Using k value of %f" %k)

write_dataset(label_with_hmm(test, train, k=k), output)
