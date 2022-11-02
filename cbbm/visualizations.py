import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
import pandas as pd


def plot_pca_meshgrid(reduced_data, init, n_init, n_clusters):
    # reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def plot_3d(data, labels_col:str):
    """
    Plot in 3 dimensions
    """
    labels = [l for l in data.loc[:,labels_col]]
    data_cols = [c for c in data.columns]

    if len(data_cols) == 2:

        print("Plotting 2 dimensions")
        df_pca = data

        fig = plt.figure() # figsize = (20, 14))

    elif len(data_cols) == 3:
        df_pca = data

        fig = plt.figure() # figsize = (20, 14))
        ax = plt.axes(projection ="3d")       

    elif len(data_cols) > 3:

        data_cols.remove(labels_col)
        data = data.loc[:,data_cols].to_numpy()

        df_pca = PCA(n_components=3).fit_transform(data)
        df_ent = pd.DataFrame(labels, columns=[labels_col])
        df_pca = pd.DataFrame(df_pca, columns=["pca1", "pca2", "pca3"])
        df_pca = pd.concat([df_ent, df_pca], axis=1)

        fig = plt.figure() # figsize = (20, 14))
        ax = plt.axes(projection ="3d")

    else:
        print("Dimensional error")
        return None

    for k in range(len(df_pca)):

        pca1 = df_pca["pca1"].loc[k]
        pca2 = df_pca["pca2"].loc[k]
        pca3 = df_pca["pca3"].loc[k]
        label = df_pca[labels_col].loc[k]
        
        ax.plot(
                    pca1,
                    pca2,
                    pca3,
                    "o",
                    markerfacecolor="red",
                    markeredgecolor="black",
                    markersize=10,
                )

        if len(data_cols) == 2:
            ax.text(pca1, pca2, label, ha="center")

        else:
            ax.text(pca1, pca2, pca3, label, ha="center")

    plt.title("PCA 3D (except if df dimensiones=2)", fontsize=10)
    fig = plt.gcf()
    plt.show()
    fig.clf()
