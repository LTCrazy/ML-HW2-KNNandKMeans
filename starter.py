import math
from collections import Counter
import numpy as np

# returns Euclidean distance between vectors a dn b
def euclidean(a, b):
    # print('a:', a)
    # print('b:', b)
    a = list(map(int, a))
    b = list(map(int, b))
    summ = sum((u - v) ** 2 for u, v in zip(a, b))
    dist = math.sqrt(summ)
    return dist


# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    dot = sum(u * v for u, v in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    dist = dot / (mag_a * mag_b)
    return dist


'test 2 dist function'
def unit_test(a, b):
    euc = euclidean(a, b)
    cos = cosim(a, b)
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.metrics.pairwise import cosine_similarity
    sk_euc = euclidean_distances([a], [b])
    print('Euclidean match sk-learn') if sk_euc == euc else 'Euclidean doesnt match'
    sk_cos = cosine_similarity([a], [b])
    print('Cos similarity match sk-learn') if sk_cos == cos else 'Cos similarity doesnt match'


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    k = 2   # hyper-parameter, could tone
    labels = []
    for query_dat in query:
        query_pt = query_dat[1]
        tuple_lst = []  # the list of tuple (dist, label)
        for train_dat in train:
            train_pt = train_dat[1]
            # print('train_pt:', train_pt, ', query_pt:', query_pt)
            dist = euclidean(train_pt, query_pt) if metric == 'euclidean' \
                else cosim(train_pt, query_pt)
            tuple_lst.append((dist, train_dat[0]))
        tuple_lst.sort()
        # Find closest point and take the majority vote
        # print('KNN\'s labels:', [x[1] for x in tuple_lst[:k]])
        vote = Counter([x[1] for x in tuple_lst[:k]]).most_common()[0][0]
        labels.append(vote)
    return labels


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train, query, metric):
    k = 10
    tol = 0.001
    max_iter = 300
    centroids = np.array([])
    classification = {}
    check = True
    # pick centroid
    centroids = np.empty([0, len(train[0])])
    for i in range(k):
        centroids = np.vstack((centroids, list(map(int, train[i]))))
        classification[i] = np.empty([0, len(train[0])])
    while check:
        # iterate through train points
        for pt in train:
            dist = []
            for i in range(len(centroids)):
                if metric == 'euclidean':
                    dist.append(euclidean(pt, centroids[i]))
                else:
                    dist.append(cosim(pt, centroids[i]))
            # print('classification:', classification)
            classification[dist.index(min(dist))] = np.vstack(
                    (classification[dist.index(min(dist))], pt))

        # recalculate centroids and break condition
        new_centroids = np.empty([0, len(train[0])])
        for i in range(k):
            new_centroid = np.average(cla2ssification[i].astype(np.int), axis=0)
            # print('new centroid:', new_centroid.shape)
            # print(classification[i].astype(np.int).shape)
            # break
            new_centroids = np.append(new_centroids, new_centroid)
            check = sum(abs((centroids[i] - new_centroid)/(centroids[i]+0.001)*100.0)) > tol
            centroids[i] = new_centroids[i]
            # empty current assignments
            classification[i] = np.empty([0, len(train[0])])

    # predict
    labels = []
    for pt in query:
        min_dist = float('inf')
        label = -1
        for i in range(len(centroids)):
            if metric == 'euclidean':
                dist = euclidean(pt, centroids[i])
            else:
                dist = cosim(pt, centroids[i])
            if dist < min_dist:
                min_dist = dist
                label = i
        labels.append(label)
    return labels


'Output 200x2x784 matrix'
def read_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return data_set


def show(file_name, mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')


def main():
    # show('valid.csv', 'pixels')
    dat_train = read_data('train.csv')
    dat_val = read_data('valid.csv')
    dat_test = read_data('test.csv')
    '''Output 200x2x784 matrix'''
    # ------------- test parameters -------------
    function = 'kmeans'
    metric = 'euclidean'
    # -------------------------------------------
    if function == 'knn':
        pred = knn(dat_train, dat_test, metric)
    elif function == 'kmeans':
        train = [x[1] for x in dat_train]
        valid = [x[1] for x in dat_val]
        test = [x[1] for x in dat_test]
        pred = kmeans(train, test, metric)
    print('predictions:', pred)
    print('labels:', [x[0] for x in dat_test])
    # Calculate accuracy
    correct = 0
    for i in range(len(dat_test)):
        correct = correct + (str(pred[i]) == dat_test[i][0])
    acc = correct / len(dat_test)
    print('accuracy:', acc)

if __name__ == "__main__":
    main()
