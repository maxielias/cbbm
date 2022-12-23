import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transform_functions import get_mean_pivot_table

class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

def fit(self, X):
        if not self.q:
            self.q = X.shape[1]
        X = np.array(X)
        sc = StandardScaler()
       # X = sc.fit_transform(X)

pfa = PFA()


pca = PCA(n_components=pfa.q).fit(X) # calculation Cov matrix is embeded in PCA
        A_q = pca.components_.T
kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_
dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))
self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]
#Run exhaustive search to find the best features to keep:
def get_key(val,y):
    for key, value in Counter(y).items():
         if val == value:
             return key
 
    return "key doesn't exist"
p = 1
#n will be the number of features you have
n = 50
for k in range(2,n,1):
    print('########## K = ',k,' #######')
    pfa = PFA(n_features=k)
    pfa.fit(df)
    # To get the transformed matrix
    x = pfa.features_
    rng = np.random.RandomState(42)
    clf = IsolationForest(n_estimators=1000, random_state=rng)
    clf.fit(x)
    IF_labels = clf.predict(x)
    m = get_key(max(Counter(IF_labels).values()),IF_labels)
    x = pd.DataFrame(x)
    x['outliers'] = IF_labels
    x = x[x['outliers'] == m ]
    x = x.drop(columns = 'outliers')
    x = np.array(x)
    pca.fit(x)
    x = pca.transform(x)
    model_agg_clustering = AgglomerativeClustering(n_clusters=3)
    # fit model and predict clusters
    yhat_agg_clustering = model_agg_clustering.fit_predict(x)
    print('agglomerative')
    model_birch = Birch(threshold=0.01, n_clusters=3)
    # fit the model
    model_birch.fit(x)
    # assign a cluster to each example
    yhat_birch = model_birch.predict(x)
    model_mb_km = MiniBatchKMeans(n_clusters=3)
    # fit the model
    model_mb_km.fit(x)
    # assign a cluster to each example
    print('Mini Batch Kmeans')
    yhat_mb_km = model_mb_km.predict(x)
    label_list = [yhat_agg_clustering, yhat_birch, yhat_mb_km]
    models = ['AgglomerativeClustering', 'BIRCH', 'MiniBatch_KMeans']
    for idx,labels in enumerate(label_list):
        silhouette_avg = silhouette_score(x, labels)
        print("The average silhouette_score for", models[idx], " is :", silhouette_avg)