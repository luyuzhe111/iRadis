import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

ckpt_dir = '../ckpt/version_0/vgg16_bn_adam_20220318012236'
acts = np.load(f'{ckpt_dir}/acts.npy')
print(acts.shape)

scaler = StandardScaler()
scale = scaler.fit_transform(acts)

pca = PCA(n_components=2)
acts_pca = pca.fit_transform(scale)

df = pd.DataFrame(acts_pca, columns=['x', 'y'])
df.to_csv(f'{ckpt_dir}/pca_acts.csv')
np.save(f'{ckpt_dir}/pca_acts.npy', acts_pca)

print(pca.explained_variance_ratio_)

start = 2
end = 10
range_n_clusters = range(start, end)
silhouette_avg = []
for n_cluster in range_n_clusters:
    # initialise kmeans
    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(acts_pca)
    cluster_labels = kmeans.labels_

    # silhouette score
    silhouette_avg.append(silhouette_score(acts_pca, cluster_labels))

opt_n = np.argmax(np.array(silhouette_avg)) + start

kmeans = KMeans(n_clusters=opt_n, random_state=0).fit(acts_pca)
labels = kmeans.labels_

plt.scatter(acts_pca[:, 0], acts_pca[:, 1], c=labels)
plt.show()

print(len(labels))
np.save(f'{ckpt_dir}/labels.npy', np.array(labels))