# CryptoClustering

![image](https://github.com/dspataru/CryptoClustering/assets/61765352/2f54a6c8-8262-4622-9ae2-6579f34b71af)


## Table of Contents
* [Background](https://github.com/dspataru/CryptoClustering/blob/main/README.md#background)
* [Preparing the Data](https://github.com/dspataru/CryptoClustering/blob/main/README.md#preparing-the-data)
* [Find the Best Value for k Using the Original Scaled DataFrame](https://github.com/dspataru/CryptoClustering/blob/main/README.md#find-the-best-value-for-k-using-the-original-scaled-dataframe)
* [Cluster Cryptocurrencies with K-means Using the Original Scaled Data](https://github.com/dspataru/CryptoClustering/blob/main/README.md#cluster-cryptocurrencies-with-k-means-using-the-original-scaled-data)
* [Optimize Clusters with Principal Component Analysis](https://github.com/dspataru/CryptoClustering/blob/main/README.md#optimize-clusters-with-principal-component-analysis)
* [Find the Best Value for k Using the PCA Data](https://github.com/dspataru/CryptoClustering/blob/main/README.md#find-the-best-value-for-k-using-the-pca-data)
* [Cluster Cryptocurrencies with K-means Using the PCA Data](https://github.com/dspataru/CryptoClustering/blob/main/README.md#cluster-cryptocurrencies-with-k-means-using-the-pca-data)


## Background

In the dynamic realm of cryptocurrencies, understanding their price movements and identifying the factors that influence them is of paramount importance. With the ever-increasing popularity and volatility of digital assets, the ability to predict whether these currencies are significantly impacted by short-term (24-hour) or longer-term (7-day) price changes is a crucial endeavor. In this assignment, we delve into the world of unsupervised learning, leveraging the power of Python, to develop a predictive model that will unravel the mysteries of cryptocurrency price fluctuations. By exploring historical data and employing cutting-edge machine learning techniques, we aim to shed light on the temporal dynamics of cryptocurrency markets, offering valuable insights for investors, traders, and enthusiasts alike.

The goal of this project is to cluster similar CryptoCurrencies together.

#### Key Words
Jupyter Notebook, machine learning, scikit-learn, KMeans clustering, Principle component analysis, PCA, dimensionality reduction, clustering algorithms, pandas, numpy, matplotlib, elbow plot


## Preparing the Data

For this project, we are using the [crypto_market_data.csv](https://github.com/dspataru/CryptoClustering/blob/main/Resources) file that contains price change information over different periods of time of 41 CryptoCurrencies. The first step was to import the csv file and review the data. In order to view the data, we plotted the price change for each cryptocurrency.

![market_data_plot](https://github.com/dspataru/CryptoClustering/blob/main/images/market_data_plot.png)

Since the data is not normalized, StandardScalar() was used from the scikit-learn library to normalize the data. The scaled data was then put in a DataFrame and the coin ID was set as the index of the DataFrame. The following is a screenshot of the DataFrame:

![scaled_df](https://github.com/dspataru/CryptoClustering/assets/61765352/79b1e761-06f5-4adf-9cb0-2e58f66f50e3)


## Find the Best Value for k Using the Original Scaled DataFrame

The elbow method was used to find the ideal number of clusters, k, to apply the KMeans clustering algorithm for clustering the data. The following steps were taken to find the best value of k:
* Create a list with the number of k values from 1 to 11.
* Create an empty list to store the inertia values.
* Create a for loop to compute the inertia with each possible value of k.
* Create a dictionary with the data to plot the elbow curve.
* Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.

![elbow curve no pca](https://github.com/dspataru/CryptoClustering/blob/main/images/elbow_curve_without_PCA.png)

Looking at the elbow curve, it looks like the best value of k is 4, meaning that the optimal number of clusters is 4. Now that we have the optimal number of clusters, we can cluster the cryptocurrencies with KMeans using the normalized data.


## Cluster Cryptocurrencies with K-means Using the Original Scaled Data

The following steps were taken to create four clusters for the cryptocurrency data:
* Initialize the K-means model with the best value for k.
* Fit the K-means model using the original scaled DataFrame.
* Predict the clusters to group the cryptocurrencies using the original scaled DataFrame.
* Create a copy of the original data and add a new column with the predicted clusters.

After the dataframe was created, hvPlot was used to plot the clusters. The x-axis was set as "price_change_percentage_24h" and the y-axis as "price_change_percentage_7d". The clusters were coloured based on the predicted clusters found using K-means. We added the "coin_id" column in the hover_cols parameter to identify the cryptocurrency represented by each data point. 

![Kmeans_cluster](https://github.com/dspataru/CryptoClustering/blob/main/images/KMeans_cluster_without_PCA.png)


## Optimize Clusters with Principal Component Analysis
Next, we optimized the clusters by performing PCA to reduce the dimensionality of the data, aka reduce the features to three principal components. Using the fit_transform function, we fit our data to the PCA model and explored the variance to determine how much information was attributed to each principal component using the explained_variance_ratio_ function on the model. PC (Principle component) 1 explains ~37% of the variance in the data, PC2 explains ~35% of the data, and PC3 explains ~18% of the data. Therefore the total explained variance is ~90%.

The first five rows of the PCA DataFrame appears as follows:

![image](https://github.com/dspataru/CryptoClustering/assets/61765352/7d52ef06-0798-4ee1-bf0c-d725b69c5616)


## Find the Best Value for k Using the PCA Data
Similar to what was done for the scaled raw data, we used the same steps for the elbow method on the PCA data to find the best value for k (number of clusters). The best value for k when using the PCA data seems to be 4 as well. Below is the elbow graph for the PCA model.

![PCA_eblow_curve](https://github.com/dspataru/CryptoClustering/blob/main/images/PCA_eblow_curve.png)

The inertia value itself is smaller for similar values of k. For example, k=4 for the PCA model is about 50, while the value for the KMeans model is around 79. However, the k-value is not different.


## Cluster Cryptocurrencies with K-means Using the PCA Data
Using the optimal number of clusters found in the previous section, we clustered the PCA data using KMeans and 4 clusters using the same steps as before. The resulting clusters are seen below.

![clusters_PCA](https://github.com/dspataru/CryptoClustering/blob/main/images/clusters_PCA.png)

We use PCA to retain all of the important information from the features. In this case, we use three principle components to best describe the highest variance in our data. PCA is a great technique to reduce the dimensionality of our data in order to be able to cluster data more efficiently and to be able to visualize the clusters. Without the dimensionality reduction, it's difficult to understand/visualize what the clusters are because there are too many features to plot. Below is the 3D plot of the clusters using the three PCs to better visualize the clusters.

![clusters_PCA](https://github.com/dspataru/CryptoClustering/blob/main/images/clusters_PCA.png)
