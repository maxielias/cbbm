import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, Normalizer, QuantileTransformer
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from kneed import KneeLocator, DataGenerator
from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
import seaborn as sns
import matplotlib.cm as cm


def find_and_drop_correlated_features(data, labels_col:str, threshold:float, drop:bool=False, plot:bool=False):
    """
    Find correlated features in dataframe.
    Optional: drop features.
    Optional: plot heatmap.
    """
    df_labels = data["ent"]
    data_cols = [c for c in data.columns]
    data_cols.remove(labels_col)
    df = data.loc[:,data_cols]

    df_corr = df.corr()

    if plot:
        sns.heatmap(df_corr)

    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    
    for i in range(df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[0]):
            if abs(df_corr.iloc[i,j]) >= threshold:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]
    
    if drop:
        data = pd.concat([df_labels, data[selected_columns]], axis=1)
        print(f"The following columns were dropped: {[d for d in data_cols if d not in selected_columns]}")

    plt.clf()
    return df_corr, data


def benchmark_scaling_and_oultlier_methods(data, labels_col:str, plot:bool=False) -> list:
    """
    ----------------------------------------------------------------
    Test several scaling methods and outlier detection algorithms
    to see which ones return certain percentage of outliers.
    ----------------------------------------------------------------
    Loop through selected scaling methods and anomaly detection algorithms provided
    inside the function.
    ----------------------------------------------------------------
    Variables:
    ----------------------------------------------------------------
    data : numpy array
    labels : labels to be used for plotting
    plot : choose if function plots all graphics (default=False)
    """
    labels = [l for l in data.loc[:,labels_col]]
    data_cols = [c for c in data.columns]
    data_cols.remove(labels_col)

    data = data.loc[:,data_cols].to_numpy()

    # More scaling methods can be added to be tested
    scaling_methods = [
        ("Standard Scaler", StandardScaler()),
        ("MaxAbs Scaler", MaxAbsScaler()),
        ("Normalizer", Normalizer()),
        ("Power Transformer", PowerTransformer(method="yeo-johnson")),
        ("Robust Scaler", RobustScaler(with_scaling=False)),
        ("Quantile Transformer", QuantileTransformer(output_distribution="normal", n_quantiles=10)),
    ]

    # More anomaly_algorithms can be added to be tested
    anomaly_algorithms = [
        ("Empirical Covariance", EllipticEnvelope(support_fraction=1.0, contamination=0.1)),
        ("Robust Covariance", EllipticEnvelope(contamination=0.1)),
        ("One-Class SVM", svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)),
        (
            "One-Class SVM (SGD)",
            make_pipeline(
                Nystroem(gamma=0.1, random_state=42, n_components=data.shape[1]),
                SGDOneClassSVM(
                    nu=0.1,
                    shuffle=True,
                    fit_intercept=True,
                    random_state=42,
                    tol=1e-6,
                ),
            ),
        ),
        (
            "Isolation Forest",
            IsolationForest(random_state=42),
        ),
        (
            "Local Outlier Factor",
            LocalOutlierFactor(contamination=0.1, n_neighbors=30),
        ),
        (
            "DBSCAN",
            DBSCAN(),
        )
    ]


    plt.figure(figsize=(len(anomaly_algorithms) * 2 + 4, len(scaling_methods) * 2 + 4))# 12.5))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    )

    scaling_and_outlier_methods_list = list()

    plot_num = 1

    for scaler in scaling_methods:
        
        algorithm_list = list()
        
        # Scale dataset
        data = scaler[1].fit_transform(data)

        # Calculate eps for DBSCAN
        knn = NearestNeighbors(n_neighbors=30)
        knn_fit = knn.fit(data)
        distances, indices = knn_fit.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        kneedle = KneeLocator(range(0, len(indices)), distances, S=1.0, curve="convex", direction="increasing")
        x_idx = kneedle.knee
        eps = distances[x_idx]
        
        # Apply PCA to plot dataset afterwards 
        X = PCA(n_components=2).fit_transform(data)

        # Maximum and minimum points needed for drawing the mesh and axis
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        for name, algorithm in anomaly_algorithms:
            '''t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()'''
            
            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(X)
            elif name == "DBSCAN":
                params = {
                    'min_samples': round(data.shape[0]*0.8), 
                    'eps': eps,
                }
                algorithm.set_params(**params)
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.fit(X).predict(X)

            
            outliers_count= list(y_pred).count(-1)
            
            # If number of outliers is between low and high boundary % append to list
            if data.shape[0]*0.1<outliers_count<data.shape[0]*0.3:
                algorithm_list.append(algorithm)

            if plot==True:

                # Step size of the mesh. Decrease to increase the quality of the VQ.
                # h = 0.01

                # Plot the decision boundary. For that, we will assign a color to each
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
                
                plt.subplot(len(scaling_methods), len(anomaly_algorithms), plot_num)
                plt.title(f"{scaler[0]}-{name}", size=10)

                # plot the levels lines and the points
                if name != "Local Outlier Factor" and name != "DBSCAN":  # LOF does not implement predict
                    Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="black")

                colors = np.array(["#377eb8", "#ff7f00"])
                plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.xticks(())
                plt.yticks(())
                '''plt.text(
                    0.99,
                    0.01,
                    ("%.2fs" % (t1 - t0)).lstrip("0"),
                    transform=plt.gca().transAxes,
                    size=15,
                    horizontalalignment="right",
                )'''
                
                if labels:
                    for i, ent in enumerate(labels):
                        if name=="DBSCAN" and y_pred[i]!=0:
                            plt.annotate(ent, (X[i,0], X[i,1]))
                        if y_pred[i]==-1: # and not name=="One-Class SVM":
                            plt.annotate(ent, (X[i,0], X[i,1]))
                        pass
                
                plot_num += 1

        scaling_and_outlier_methods_list.append([scaler[1], algorithm_list])

    if plot==True:
        plt.show()
        plt.clf()

    return scaling_and_outlier_methods_list


def drop_outliers(data, labels_col:str, scaler:int, algorithm:int, drop:bool=True, params=None):
    """
    ----------------------------------------------------------------
    Drop outliers in a dataset.
    ----------------------------------------------------------------
    Select the scaling method and anomaly detection algorithm from the ones provided
    in the description.
    ----------------------------------------------------------------
    Variables:
    ----------------------------------------------------------------
    data : dataframe of shape m,n which includes a column with labels\n
    labels : name of the column to be set as the labels list\n
    scaler : choose one of the scaling methods listed below\n
    1) Standard Scaler
    2) MaxAbs Scaler
    3) Normalizer
    4) Power Transformer
    5) Robust Scaler
    6) Quantile Transformer\n
    method : for finding outliers\n
    1) Standard Scaler
    2) MaxAbs Scaler
    3) Normalizer
    4) Power Transformer
    5) Robust Scaler
    6) Quantile Transformer\n
    params: set to None for the moment

    """
    labels = [l for l in data.loc[:,labels_col]]
    data_cols = [c for c in data.columns]
    data_cols.remove(labels_col)

    data = data.loc[:,data_cols].to_numpy()

    if not params:
        params=None

    # More scaling methods can be added to be tested
    scaling_methods = [
        ("Standard Scaler", StandardScaler()),
        ("MaxAbs Scaler", MaxAbsScaler()),
        ("Normalizer", Normalizer(norm="max")),
        ("Power Transformer", PowerTransformer(method="yeo-johnson")),
        ("Robust Scaler", RobustScaler(with_scaling=False)),
        ("Quantile Transformer", QuantileTransformer(output_distribution="normal", n_quantiles=10)),
    ]

    # More anomaly_algorithms can be added to be tested
    anomaly_algorithms = [
        ("Empirical Covariance", EllipticEnvelope(support_fraction=1.0, contamination=0.1)),
        ("Robust Covariance", EllipticEnvelope(contamination=0.1)),
        ("One-Class SVM", svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)),
        (
            "One-Class SVM (SGD)",
            make_pipeline(
                Nystroem(gamma=0.1, random_state=42, n_components=data.shape[1]),
                SGDOneClassSVM(
                    nu=0.1,
                    shuffle=True,
                    fit_intercept=True,
                    random_state=42,
                    tol=1e-6,
                ),
            ),
        ),
        (
            "Isolation Forest",
            IsolationForest(random_state=42),
        ),
        (
            "Local Outlier Factor",
            LocalOutlierFactor(contamination=0.1, n_neighbors=30),
        ),
        '''(
            "DBSCAN",
            DBSCAN(),
        )'''
    ]

    if scaler>0:
        scaler = scaling_methods[scaler-1][1]
        # Scale the data
        data = scaler.fit_transform(data)

    name, anomaly_algorithm = anomaly_algorithms[algorithm-1]

    print(scaler, name)

    # fit the data and tag outliers
    if name == "Local Outlier Factor":
        y_pred = anomaly_algorithm.fit_predict(data)
    elif name == "DBSCAN":
        # Calculate eps for DBSCAN
        knn = NearestNeighbors(n_neighbors=30)
        knn_fit = knn.fit(data)
        distances, indices = knn_fit.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        kneedle = KneeLocator(range(0, len(indices)), distances, S=1.0, curve="convex", direction="increasing")
        x_idx = kneedle.knee
        eps = distances[x_idx]
        params = {
            'min_samples': round(data.shape[0]*0.9), 
            'eps': eps,
        }
        anomaly_algorithm.set_params(**params)
        y_pred = anomaly_algorithm.fit_predict(data)
    else:
        y_pred = anomaly_algorithm.fit(data).predict(data)

    
    inliers_list_idx = [i for i, e in enumerate(list(y_pred)) if e==1]
    outliers_list_idx = [i for i, e in enumerate(list(y_pred)) if e==-1]

    df_data = pd.DataFrame(data, columns=data_cols)
    df_labels = pd.DataFrame(labels, columns=[labels_col])
    # df_pred = pd.DataFrame(list(y_pred), columns=["outlier"])
    df = pd.concat([df_labels, df_data], axis=1)
    
    if drop==True:
        try:
            print(f"Droppped the following samples: {list(df[labels_col].iloc[outliers_list_idx])}")
            df = df.iloc[inliers_list_idx]
        except Exception as e:
            print(e)
            print("Error: no outliers dropped")
            pass
    
    df = df.reset_index().drop(columns="index")

    return df


def optimal_k(data, labels_col:str, model:str, params:dict, max_k:int, scaler:int=0, drop:bool=False, algorithm:int=0, plot:bool=False):
    """

    -------------------------------------------------------------------
    Find optimal number of clusters with elbow method.\n
    It iterates over a range of clusters given as input.\n
    -------------------------------------------------------------------
    data : dataframe of shape m,n which includes a column with labels\n
    labels_col : name of the column to be set as the labels list\n
    model: select clustering method. Suggested:\n
    \t- "kmeans"
    \t- "kmedoids"
    \t- "minibatch_kmeans"
    \t- "dbscan"
    \t- "gaussianmixture"\n
    params : dict of params to be used in the model.\n
    max_k : maximum number of clusters to be processed.\n
    scaler : choose one of the scaling methods listed below (default=0)\n
    0) None
    1) Standard Scaler
    2) MaxAbs Scaler
    3) Normalizer
    4) Power Transformer
    5) Robust Scaler
    6) Quantile Transformer\n
    """
    
    if drop:
        data = drop_outliers(data, labels_col=labels_col, scaler=scaler, algorithm=algorithm, drop=True)
    
    labels = [l for l in data.loc[:,labels_col]]
    data_cols = [c for c in data.columns]
    data_cols.remove(labels_col)

    data = data.loc[:,data_cols].to_numpy()

    n_clusters = range(2, max_k+1)

    # More scaling methods can be added to be tested
    scaling_methods = [
        ("Standard Scaler", StandardScaler()),
        ("MaxAbs Scaler", MaxAbsScaler()),
        ("Normalizer", Normalizer()),
        ("Power Transformer", PowerTransformer(method="yeo-johnson")),
        ("Robust Scaler", RobustScaler(with_scaling=False)),
        ("Quantile Transformer", QuantileTransformer(output_distribution="normal", n_quantiles=10)),
    ]

    anomaly_algorithms = [
        ("Empirical Covariance", EllipticEnvelope(support_fraction=1.0, contamination=0.1)),
        ("Robust Covariance", EllipticEnvelope(contamination=0.1)),
        ("One-Class SVM", svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)),
        (
            "One-Class SVM (SGD)",
            make_pipeline(
                Nystroem(gamma=0.1, random_state=42, n_components=data.shape[1]),
                SGDOneClassSVM(
                    nu=0.1,
                    shuffle=True,
                    fit_intercept=True,
                    random_state=42,
                    tol=1e-6,
                ),
            ),
        ),
        (
            "Isolation Forest",
            IsolationForest(random_state=42),
        ),
        (
            "Local Outlier Factor",
            LocalOutlierFactor(contamination=0.1, n_neighbors=30),
        ),
        '''(
            "DBSCAN",
            DBSCAN(),
        )'''
    ]

    # metric titles
    report_columns = [
        "method",
        "k",
        "inertia",
        "homogeneity_score",
        "completeness_score",
        "v_measure_score",
        "adjusted_rand_score",
        "adjusted_mutual_info_score",
        "silhouette_score",
        "silhouette_samples",
        "davies_bouldin_score",
        "calinski_harabasz_score"
    ]

    model_dict = {
        "kmeans": ("KMeans", KMeans()),
        "kmedoids": ("KMedoids", KMedoids()),
        "minibatch_kmeans": ("MiniBatchKMeans", MiniBatchKMeans()),
    }

    df = pd.DataFrame([], columns=report_columns)

    for n in n_clusters:

        params["n_clusters"] = n

        model_pipeline = model_dict[model][1]
        model_name = model_dict[model][0]

        model_pipeline = model_pipeline.set_params(**params)

        if scaler==0 or drop==True:
            try:
                scaler_name = "None"
                estimator = make_pipeline(model_pipeline).fit(data)
                # predict_estimator = make_pipeline(model_pipeline).predict(data)

            except Exception as e:
                print(e)
                print("Too many outliers.")
                return df

        else:
            scaler_name = scaling_methods[scaler-1][0]
            scaler_pipeline = scaling_methods[scaler-1][1]
            estimator = make_pipeline(scaler_pipeline, model_pipeline).fit(data)

        # print(estimator[-1].labels_)
        # print(estimator[-1].cluster_centers_)
        results = [f'{model_name}(k={n})-{scaler_name}', n, estimator[-1].inertia_]

        # Define the metrics which require only the true labels and estimator
        # labels
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.v_measure_score,
            metrics.adjusted_rand_score,
            metrics.adjusted_mutual_info_score,
        ]
        results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

        # The silhouette score requires the full dataset
        try:
            results += [
                metrics.silhouette_score(
                    data,
                    estimator[-1].labels_,
                    metric="euclidean",
                    sample_size=300,
                ),
                metrics.silhouette_samples(
                    data,
                    estimator[-1].labels_
                )
            ]
        except Exception as e:
            results += [0, 0]
            print(e)
            continue

        results += [
            metrics.davies_bouldin_score(
                data,
                estimator[-1].labels_,
            )
        ]

        results += [
            metrics.calinski_harabasz_score(
                data,
                estimator[-1].labels_,
            )
        ]

        df_length = len(df)
        df.loc[df_length] = results

    silhouette_optimal_k = df["k"].loc[df["silhouette_score"].idxmax()]
    calinski_harabasz_optimal_k = df["k"].loc[df["calinski_harabasz_score"].idxmax()]

    if plot:

        kelbow_model = KElbowVisualizer(model_pipeline, k=(2, max_k))
        kelbow_model.fit(data)
        plt.show()
        plt.clf()

        elbow = kelbow_model.elbow_value_

        predicted_k = pd.DataFrame({"sample":labels})

        for tup in [("Elbow Method", elbow), ("Silhouette Method", silhouette_optimal_k), ("Calinski Harabsz Method", calinski_harabasz_optimal_k)]: 

            k_optimal = tup[1]
            method_name = tup[0]
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(data) + (k_optimal + 1) * 10])

            # Initialize the clusterer with n_clusters value
            # and a random generator seed of 10 for reproducibility.

            params["n_clusters"] = k_optimal
            model_pipeline = model_dict[model][1]
            model_name = model_dict[model][0]
            model_pipeline = model_pipeline.set_params(**params)
            
            if scaler==0:
                estimator = make_pipeline(model_pipeline).fit(data)

            else:
                scaler_name = scaling_methods[scaler-1][0]
                scaler_pipeline = scaling_methods[scaler-1][1]
                estimator = make_pipeline(scaler_pipeline, model_pipeline).fit(data)

            predict_k = estimator.predict(data)
            predicted_k = pd.concat([predicted_k, pd.DataFrame({method_name:predict_k})], axis=1)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters

            silhouette_avg = df["silhouette_score"].loc[df["k"]==k_optimal].item()
            # print(silhouette_avg)
            '''print(
                "For n_clusters =",
                k_optimal,
                "The average silhouette_score is :",
                silhouette_avg,
            )'''

            # Compute the silhouette scores for each sample
            sample_silhouette_values = df["silhouette_samples"].loc[df["k"]==k_optimal].item()

            y_lower = 10

            list_colors = list()

            for i in range(k_optimal):
                
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[estimator[-1].labels_==i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / k_optimal)
                list_colors.append(color)

                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )
                
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([r/10 for r in range(-10,10,2)])

            data_pca = PCA(n_components=2).fit_transform(data)
            
            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(estimator[-1].labels_.astype(float) / k_optimal)
            ax2.scatter(
                data_pca[:, 0], data_pca[:, 1], marker=".", s=30, lw=1, alpha=0.7, c=colors, edgecolor=colors
            )

            # Labeling the clusters
            centers = estimator[-1].cluster_centers_
            # print(centers)

            centers_pca = PCA(n_components=2).fit_transform(centers)

            # Draw white circles at cluster centers
            ax2.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                marker="o",
                color=list_colors,
                alpha=1,
                s=200,
                edgecolor=list_colors,
            )

            for i, c in enumerate(centers_pca):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=75, color="w", edgecolor="w")
                ax2.annotate(f"{round(c[0], 2)}, {round(c[1], 2)}", (c[0]+0.07, c[1]+0.07))

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                f"Silhouette analysis of {method_name} for {model_name} with {k_optimal} clusters.\n Sample data scaled with {scaler_name} using {anomaly_algorithms[algorithm][0]} for outlier detection",
                fontsize=14,
                fontweight="bold",
            )

            plt.savefig(f"graph/{method_name}-{model_name}-{k_optimal}-{scaler_name}-{anomaly_algorithms[algorithm][0]}.png") # , bbox_inches='tight')

        print(predicted_k)
        predicted_k.to_csv(f"output/{model_name}-{k_optimal}-{scaler_name}-{anomaly_algorithms[algorithm][0]}.csv", sep=",", index=False)
        
        plt.show()
        plt.clf()

        print(f"Elbow Method optimal K is {elbow}")

    print(f"Silhouette Method optimal K is {silhouette_optimal_k}")
    print(f"Calinski Harabasz Method optimal K is {calinski_harabasz_optimal_k}")
    
    return df


'''if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_digits

    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

    print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    bench_k_means_test(kmeans=kmeans, name="k-means++", data=data, labels=labels)

    kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
    bench_k_means_test(kmeans=kmeans, name="random", data=data, labels=labels)

    pca = PCA(n_components=n_digits).fit(data)
    kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
    bench_k_means_test(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

    print(82 * "_")'''