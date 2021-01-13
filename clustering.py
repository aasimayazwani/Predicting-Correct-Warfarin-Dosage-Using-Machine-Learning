def values(df2):
	from sklearn.preprocessing import normalize
	col = df2.columns.values
	data = pd.DataFrame(normalize(df2),columns = col)
    from sklearn.cluster import KMeans
    kmeans=KMeans(n_clusters=2)
    kmeans = pd.DataFrame(kmeans.fit_predict(df2),columns = ["kmeans"])

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=2)
    GMM = pd.DataFrame(gmm.fit_predict(df2),columns = ["Gaussian_Mixture_Models"])

    from sklearn.cluster import AgglomerativeClustering
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    Agglomerative_results = pd.DataFrame(cluster.fit_predict(df2),columns = ["Agglomerative_Clustering"])

    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    hdb = pd.DataFrame(clusterer.fit_predict(df2),columns = ["HDensityBased"])
    s = pd.concat([Agglomerative_results, GMM, kmeans,hdb], axis = 1)
    return s 
some = values(data)


from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
col = data.columns.values
data = pd.DataFrame(normalize(data),columns = col)
import seaborn as sns
pca = PCA(n_components=2)
from sklearn.preprocessing import scale 
data = pd.DataFrame(scale(data),columns = col)
p = pca.fit_transform(data)
p = pd.DataFrame(p,columns =["pc1","pc2"])
k = pd.concat([p, some.Agglomerative_Clustering], axis=1, join='inner')
k = pd.concat([k, some.Gaussian_Mixture_Models], axis=1, join='inner')
k = pd.concat([k, some.kmeans], axis=1, join='inner')
k = pd.concat([k, some.HDensityBased], axis=1, join='inner')
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
sns.scatterplot("pc1","pc2",hue ="Agglomerative_Clustering",data = k,legend = "full",sizes=(40, 150))
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")





