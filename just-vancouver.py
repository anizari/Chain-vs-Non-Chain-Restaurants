# Limit OSM data to just greater Vancouver
# Typical invocation:
# spark-submit just-vancouver.py amenities amenities-vancouver
# hdfs dfs -cat amenities-vancouver/* | gzip -d - | gzip -c > amenities-vancouver.json.gz

import sys
import unicodedata
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from pyspark.ml.clustering import (LDA, BisectingKMeans, GaussianMixture,
                                   KMeans, PowerIterationClustering)
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import Row, SparkSession, functions, types

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

spark = SparkSession.builder.appName(
    'OSM point of interest extracter').getOrCreate()
assert spark.version >= '2.4'  # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
# sc = spark.sparkContext


amenity_schema = types.StructType([
    types.StructField('lat', types.DoubleType(), nullable=False),
    types.StructField('lon', types.DoubleType(), nullable=False),
    types.StructField('timestamp', types.TimestampType(), nullable=False),
    types.StructField('amenity', types.StringType(), nullable=False),
    types.StructField('name', types.StringType(), nullable=True),
    types.StructField('tags', types.MapType(
        types.StringType(), types.StringType()), nullable=False),
    types.StructField('wikidata', types.StringType(), nullable=False),
])


def filterData(poi):
    poi = poi.filter((poi['lon'] > -123.5) & (poi['lon'] < -122))
    poi = poi.filter((poi['lat'] > 49) & (poi['lat'] < 49.5))

    # Filter amenities to remove uninteresting places
    # Amenity tag values taken from https://wiki.openstreetmap.org/wiki/Key:amenity#Values
    # Can be placed into seperate function later

    # Food
    foodArray = ['bar', 'biergarten', 'cafe', 'fast_food',
                 'food_court', 'ice_cream', 'pub', 'restaurant']

    poi = poi.filter(poi['amenity'].isin(foodArray))

    vecAssembler = VectorAssembler(
        inputCols=['lat', 'lon'], outputCol="features")
    poi = vecAssembler.transform(poi).cache()

    brands = np.array(poi.select(
        poi['tags']['brand']).distinct().dropna().collect())
    brands = brands.flatten()
    brands = list(brands)

    chain = poi.filter(reduce(
        lambda a, b: a | b, (poi['name'].like('%' + pat + "%") for pat in brands)))
    # chain.show()

    nonchain = poi.filter(~reduce(
        lambda a, b: a | b, (poi['name'].like('%' + pat + "%") for pat in brands)))
    # nonchain.show()

    # for item in brands:
    #     normal_item = unicodedata.normalize('NFKD', str(item[0]))
    #     print(normal_item.encode('ASCII', 'ignore'))

    return chain, nonchain


def distance(city, stations):
    lat1 = city['latitude']
    lon1 = city['longitude']
    lat2 = stations['latitude']
    lon2 = stations['longitude']
    R = 6371
    p = np.pi / 180  # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * \
        np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    distance = (2 * R * np.arcsin(np.sqrt(a)))
    return distance


def cluster_KM(data, test_data):
    # Code from https://towardsdatascience.com/k-means-clustering-using-pyspark-on-big-data-6214beacdc8b
    # scale = StandardScaler(inputCol='features', outputCol='standardized')
    # data_scale = scale.fit(data)
    # data_scale = data_scale.transform(data)

    # evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
    #                                 metricName='silhouette', distanceMeasure='squaredEuclidean')

    # # Cost of k
    # cost = np.zeros(20)
    # for i in range(2, 20):
    #     gmm = KMeans(featuresCol='standardized', k=i)
    #     model = gmm.fit(data_scale)

    #     predictions = model.transform(data_scale)
    #     cost[i] = evaluator.evaluate(predictions)

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.plot(range(2, 20), cost[2:20])
    # ax.set_xlabel('k')
    # ax.set_ylabel('cost')
    # plt.show()

    # Code from https://spark.apache.org/docs/latest/ml-clustering.html
    # Trains a k-means model.
    kmeans = KMeans(k=13)
    model = kmeans.fit(data)

    # Make predictions
    predictions = model.transform(data)
    test_predictions = model.transform(test_data)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    test_silhouette = evaluator.evaluate(test_predictions)
    print("Silhouette with squared euclidean distance = " +
          str(silhouette) + str(test_silhouette))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    return predictions, test_predictions


def cluster_BKM(data, test_data):
    # Code from https://towardsdatascience.com/k-means-clustering-using-pyspark-on-big-data-6214beacdc8b
    # scale = StandardScaler(inputCol='features', outputCol='standardized')
    # data_scale = scale.fit(data)
    # data_scale = data_scale.transform(data)

    # evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
    #                                 metricName='silhouette', distanceMeasure='squaredEuclidean')

    # # Cost of k
    # cost = np.zeros(20)
    # for i in range(2, 20):
    #     gmm = BisectingKMeans(featuresCol='standardized', k=i)
    #     model = gmm.fit(data_scale)

    #     predictions = model.transform(data_scale)
    #     cost[i] = evaluator.evaluate(predictions)

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.plot(range(2, 20), cost[2:20])
    # ax.set_xlabel('k')
    # ax.set_ylabel('cost')
    # plt.show()

    # Code from https://spark.apache.org/docs/latest/ml-clustering.html
    # Trains a bisecting k-means model.
    bkm = BisectingKMeans(k=14)
    model = bkm.fit(data)

    # Make predictions
    predictions = model.transform(data)
    test_predictions = model.transform(test_data)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    test_silhouette = evaluator.evaluate(test_predictions)
    print("Silhouette with squared euclidean distance = " +
          str(silhouette) + str(test_silhouette))

    # Shows the result.
    print("Cluster Centers: ")
    centers = model.clusterCenters()
    for center in centers:
        print(center)

    return predictions, test_predictions


def cluster_GMM(data, test_data):
    # Code from https://towardsdatascience.com/k-means-clustering-using-pyspark-on-big-data-6214beacdc8b
    # scale = StandardScaler(inputCol='features', outputCol='standardized')
    # data_scale = scale.fit(data)
    # data_scale = data_scale.transform(data)

    # evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
    #                                 metricName='silhouette', distanceMeasure='squaredEuclidean')

    # # Cost of k
    # cost = np.zeros(20)
    # for i in range(2, 20):
    #     gmm = GaussianMixture(featuresCol='standardized', k=i)
    #     model = gmm.fit(data_scale)

    #     predictions = model.transform(data_scale)
    #     cost[i] = evaluator.evaluate(predictions)

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # ax.plot(range(2, 20), cost[2:20])
    # ax.set_xlabel('k')
    # ax.set_ylabel('cost')
    # plt.show()

    # Code from https://spark.apache.org/docs/latest/ml-clustering.html
    # Trains a bisecting gmm model.
    gmm = GaussianMixture(k=15)
    model = gmm.fit(data)

    # Make predictions
    predictions = model.transform(data)
    test_predictions = model.transform(test_data)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    test_silhouette = evaluator.evaluate(test_predictions)
    print("Silhouette with squared euclidean distance = " +
          str(silhouette) + str(test_silhouette))

    # print("Gaussians shown as a DataFrame: ")
    # model.gaussiansDF.show(truncate=False)

    return predictions, test_predictions


def ML(data, test_data):
    km_df, km_test = cluster_KM(data, test_data)
    bkm_df, bkm_test = cluster_BKM(data, test_data)
    gmm_df, gmm_test = cluster_GMM(data, test_data)

    return km_df, km_test, bkm_df, bkm_test, gmm_df, gmm_test

def pandasDataframe(chain, nonchain):
    chain_pd = chain.toPandas()
    nonchain_pd = nonchain.toPandas()

    return chain_pd, nonchain_pd

def plotKMeansFig(chain, nonchain):
    fig, (p1,p2) = plt.subplots(1,2, figsize=(16,9))
    fig.suptitle('K-Means')
    p1.scatter(chain.lat, chain.lon, c=chain.prediction)
    p1.set_title('Chain Food Locations')
    p1.set_xlabel('Latitude')
    p1.set_ylabel('Longitude')
    p2.scatter(nonchain.lat, nonchain.lon, c = nonchain.prediction)
    p2.set_title('Nonchain Food Locations')
    p2.set_xlabel('Latitude')
    p2.set_ylabel('Longitude')
    fig.savefig('figures/kmeans.png')

def plotBkmFig(chain, nonchain):
    fig, (p1,p2) = plt.subplots(1,2, figsize=(16,9))
    fig.suptitle('Bisecting KMeans')
    p1.scatter(chain.lat, chain.lon, c=chain.prediction)
    p1.set_title('Chain Food Locations')
    p1.set_xlabel('Latitude')
    p1.set_ylabel('Longitude')
    p2.scatter(nonchain.lat, nonchain.lon, c = nonchain.prediction)
    p2.set_title('Nonchain Food Locations')
    p2.set_xlabel('Latitude')
    p2.set_ylabel('Longitude')
    fig.savefig('figures/bkm.png')

def plotGmmFig(chain, nonchain):
    fig, (p1,p2) = plt.subplots(1,2, figsize=(16,9))
    fig.suptitle('Gaussian Mixture Model')
    p1.scatter(chain.lat, chain.lon, c=chain.prediction)
    p1.set_title('Chain Food Locations')
    p1.set_xlabel('Latitude')
    p1.set_ylabel('Longitude')
    p2.scatter(nonchain.lat, nonchain.lon, c = nonchain.prediction)
    p2.set_title('Nonchain Food Locations')
    p2.set_xlabel('Latitude')
    p2.set_ylabel('Longitude')
    fig.savefig('figures/gmm.png')

def main(inputs, output):
    poi = spark.read.json(inputs, schema=amenity_schema)
    chain, nonchain = filterData(poi)
    chain = chain.cache()
    nonchain = nonchain.cache()

    chain_km, nonchain_km, chain_bkm, nonchain_bkm, chain_gmm, nonchain_gmm = ML(
        chain, nonchain)

    # chain.show()
    # chain_km.show()

    #Convert pypark dataframes to panda dataframe for plotting
    chain_km_pd, nonchain_km_pd = pandasDataframe(chain_km, nonchain_km)
    chain_bkm_pd, nonchain_bkm_pd = pandasDataframe(chain_bkm, nonchain_bkm)
    chain_gmm_pd, nonchain_gmm_pd = pandasDataframe(chain_gmm, nonchain_gmm)

    seaborn.set()

    #Plot each clustering classifier

    #KMeans
    plotKMeansFig(chain_km_pd, nonchain_km_pd)

    # fig1, (p1,p2) = plt.subplots(1,2, figsize=(16,9))
    # fig1.suptitle('K-Means')
    # p1.scatter(chain_km_pd.lat, chain_km_pd.lon, c=chain_km_pd.prediction)
    # p1.set_title('Chain Food Locations')
    # p1.set_xlabel('Latitude')
    # p1.set_ylabel('Longitude')
    # p2.scatter(nonchain_km_pd.lat, nonchain_km_pd.lon, c = nonchain_km_pd.prediction)
    # p2.set_title('Nonchain Food Locations')
    # p2.set_xlabel('Latitude')
    # p2.set_ylabel('Longitude')
    # fig1.savefig('figures/kmeans.png')

    #BKM
    plotBkmFig(chain_bkm_pd, nonchain_bkm_pd)

    # fig2, (p3,p4) = plt.subplots(1,2, figsize=(16,9))
    # fig2.suptitle('Bisecting KMeans')
    # p3.scatter(chain_bkm_pd.lat, chain_bkm_pd.lon, c=chain_bkm_pd.prediction)
    # p3.set_title('Chain Food Locations')
    # p3.set_xlabel('Latitude')
    # p3.set_ylabel('Longitude')
    # p4.scatter(nonchain_bkm_pd.lat, nonchain_bkm_pd.lon, c = nonchain_bkm_pd.prediction)
    # p4.set_title('Nonchain Food Locations')
    # p4.set_xlabel('Latitude')
    # p4.set_ylabel('Longitude')
    # fig2.savefig('figures/bkm.png')

    #GMM
    plotGmmFig(chain_gmm_pd, nonchain_gmm_pd)

    # fig3, (p5,p6) = plt.subplots(1,2, figsize=(16,9))
    # fig3.suptitle('Gaussian Mixture Model')
    # p5.scatter(chain_gmm_pd.lat, chain_gmm_pd.lon, c=chain_gmm_pd.prediction)
    # p5.set_title('Chain Food Locations')
    # p5.set_xlabel('Latitude')
    # p5.set_ylabel('Longitude')
    # p6.scatter(nonchain_gmm_pd.lat, nonchain_gmm_pd.lon, c = nonchain_gmm_pd.prediction)
    # p6.set_title('Nonchain Food Locations')
    # p6.set_xlabel('Latitude')
    # p6.set_ylabel('Longitude')
    # fig3.savefig('figures/gmm.png')



    # poi.show()
    poi = poi.coalesce(1)  # ~1MB after the filtering
    poi.write.json(output, mode='overwrite', compression='gzip')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
