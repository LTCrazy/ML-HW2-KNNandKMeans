import math
from collections import Counter
import numpy as np

k = 0

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
    # print('a:', a)
    # print('b:', b)
    a = list(map(int, a))
    b = list(map(int, b))
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
    k = 2  # hyper-parameter, could tone
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
    # max_iter = 300
    centroids = np.array([])
    classification = {}
    check = True
    # pick centroid
    centroids = np.empty([0, len(train[0])])
    for i in range(k):
        centroids = np.vstack((centroids, list(map(int, train[-i]))))
        classification[i] = np.empty([0, len(train[0])])
    while check:
        # iterate through train points
        for pt in train:
            dist = []
            for i in range(k):
                if metric == 'euclidean':
                    dist.append(euclidean(pt, centroids[i]))
                else:
                    dist.append(cosim(pt, centroids[i]))
            classification[dist.index(min(dist))] = np.vstack(
                (classification[dist.index(min(dist))], pt))
        # for k, val in classification.items():
        #     print('classification', k, ' values:', len(val))
        # recalculate centroids and break condition
        # new_centroids = np.empty([0, len(train[0])])
        check = False
        for i in range(k):
            new_centroid = np.average(classification[i].astype(np.int), axis=0)
            print('Nan:', new_centroid) if np.isnan(new_centroid.any()) else ''
            # print('new centroid:', new_centroid.shape)
            # print(classification[i].astype(np.int).shape)
            # convergence test
            # new_centroids = np.append(new_centroids, new_centroid)
            if np.sum((abs(centroids[i] - new_centroid) / (centroids[i] + 0.001))) > tol:
                check = True
            # ---- only check for all convergence
            centroids[i] = new_centroid
            # empty current assignments
            classification[i] = np.empty([0, len(train[0])])

    # predict
    labels = []
    for pt in query:
        min_dist = float('inf')
        label = -1
        for i in range(k):
            if metric == 'euclidean':
                dist = euclidean(pt, centroids[i])
            else:
                dist = cosim(pt, centroids[i])
            # print('dist:', dist)
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


def dimensionality_reduction(filename):
    train_data = read_data(filename)
    train_2d = []
    train_labels = []
    for row in range(len(train_data)):
        X_train = train_data[row][1]
        X_label = train_data[row][0]
        train_2d.append(X_train)
        train_labels.append(X_label)

    pca_trans_data = PCA(n_components=100, svd_solver='randomized', whiten=True).fit(train_2d)
    X_train_pca = pca_trans_data.transform(train_2d)
    X_train = X_train_pca.astype(str).tolist()
    train = [list(e) for e in zip(train_labels, X_train)]
    pca_variance = pca_trans_data.explained_variance_ratio_.sum()
    return train


def main():
    # show('valid.csv', 'pixels')
    dat_train = read_data('train.csv')
    dat_val = read_data('valid.csv')
    dat_test = read_data('test.csv')
    '''Output 200x2x784 matrix'''
    # ------------- test parameters -------------
    function = 'kmeans'
    metric = 'cosim'
    # k = 10
    # -------------------------------------------
    if function == 'knn':
        pred = knn(dat_train, dat_test, metric)
    elif function == 'kmeans':
        train = [x[1] for x in dat_train]
        valid = [x[1] for x in dat_val]
        test = [x[1] for x in dat_test]
        pred = kmeans(train, valid, metric)
    # print('predictions:', pred)
    # print('labels:', [x[0] for x in dat_test])
    # Calculate accuracy
    # take mode(label) - confusion matrix
    # plot centroids
    correct = 0
    for i in range(len(dat_test)):
        correct = correct + (str(pred[i]) == dat_test[i][0])
    acc = float(correct) / len(dat_test)
    print('accuracy:', acc)


if __name__ == "__main__":
    main()


###Soft k means

def cluster_fn(centers, x, beta):
    N, _ = x.shape
    K, D = centers.shape
    R = np.zeros((N, K))

    for n in range(N):
        R[n] = np.exp(-beta * np.linalg.norm(centers - x[n], 2, axis=1))
    R /= R.sum(axis=1, keepdims=True)

    return R


def soft_k_means(x, k=3, max_iters=20, beta=1.):
    #Initializing centers
    N, D = x.shape
    centers = np.zeros((k, D))
    arr = []
    for i in range(k):
        idx = np.random.choice(N)
        while j in arr:
            j = np.random.choice(N)
        arr.append(j)
        centers[i] = x[j]

    prev_cost = 0

    for _ in range(max_iters):
        r = cluster_fn(centers, x, beta)

        #Updating centers
        N, D = x.shape
        centers = np.zeros((k, D))
        for i in range(k):
            centers[i] = r[:, i].dot(x) / r[:, i].sum()

        #Calculating cost
        cost = 0
        for i in range(k):
            norm = np.linalg.norm(x - centers[i], 2)
            cost += (norm * np.expand_dims(r[:, i], axis=1)).sum()

        #Break condition
        if np.abs(cost - prev_cost) < 1e-5:
            break
        prev_cost = cost