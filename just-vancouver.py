# Limit OSM data to just greater Vancouver
# Typical invocation:
# spark-submit just-vancouver.py amenities amenities-vancouver
# hdfs dfs -cat amenities-vancouver/* | gzip -d - | gzip -c > amenities-vancouver.json.gz

import sys
from functools import reduce

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, types

sns.set()

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


def filterRegions(poi):
    burnabyRegion = poi.filter(
        (poi['lon'] > -123.023746) & (poi['lon'] < -122.892596))
    burnabyRegion = poi.filter(
        (poi['lat'] > 49.201466) & (poi['lat'] < 49.293732))

    vancouverRegion = poi.filter(
        (poi['lon'] > -123.287332) & (poi['lon'] < -123.023504))
    vancouverRegion = poi.filter(
        (poi['lat'] > 49.216320) & (poi['lat'] < 49.406466))

    richmondRegion = poi.filter(
        (poi['lon'] > -123.209143) & (poi['lon'] < -123.059111))
    richmondRegion = poi.filter(
        (poi['lat'] > 49.119807) & (poi['lat'] < 49.200504))

    coquitlamRegion = poi.filter(
        (poi['lon'] > -122.893479) & (poi['lon'] < -122.727209))
    coquitlamRegion = poi.filter(
        (poi['lat'] > 49.238488) & (poi['lat'] < 49.350962))

    surreyRegion = poi.filter(
        (poi['lon'] > -122.895803) & (poi['lon'] < -122.682257))
    surreyRegion = poi.filter(
        (poi['lat'] > 49.006255) & (poi['lat'] < 49.211449))

    mapleRidgeRegion = poi.filter(
        (poi['lon'] > -122.662586) & (poi['lon'] < -122.558215))
    mapleRidgeRegion = poi.filter(
        (poi['lat'] > 49.200828) & (poi['lat'] < 49.241863))

    deltaRegion = poi.filter(((poi['lon'] > -122.934625) & (poi['lon'] < -122.891791)
                              & (poi['lat'] > 49.077707) & (poi['lat'] < 49.176530))
                             | ((poi['lon'] > -123.183959) & (poi['lon'] < -122.963632)
                                & (poi['lat'] > 49.007463) & (poi['lat'] < 49.114209)))

    langleyRegion = poi.filter(
        (poi['lon'] > -122.679302) & (poi['lon'] < -122.466271))
    langleyRegion = poi.filter(
        (poi['lat'] > 49.003587) & (poi['lat'] < 49.166102))

    abbotsfordRegion = poi.filter(
        (poi['lon'] > -122.457208) & (poi['lon'] < -122.100152))
    abbotsfordRegion = poi.filter(
        (poi['lat'] > 49.005245) & (poi['lat'] < 49.125365))

    bbyChain, bbyNonChain = filterFoodData(burnabyRegion)
    vanChain, vanNonChain = filterFoodData(vancouverRegion)
    richChain, richNonChain = filterFoodData(richmondRegion)
    coqChain, coqNonChain = filterFoodData(coquitlamRegion)
    sryChain, sryNonChain = filterFoodData(surreyRegion)
    mrChain, mrNonChain = filterFoodData(mapleRidgeRegion)
    deltaChain, deltaNonChain = filterFoodData(deltaRegion)
    lanChain, lanNonChain = filterFoodData(langleyRegion)
    abbChain, abbNonChain = filterFoodData(abbotsfordRegion)

    return bbyChain, bbyNonChain, vanChain, vanNonChain, richChain, \
        richNonChain, coqChain, coqNonChain, sryChain, sryNonChain, \
        mrChain, mrNonChain, deltaChain, deltaNonChain, lanChain, \
        lanNonChain, abbChain, abbNonChain


def filterGreaterVancouver(poi):
    poi = poi.filter((poi['lon'] > -123.5) & (poi['lon'] < -122))
    poi = poi.filter((poi['lat'] > 49) & (poi['lat'] < 49.5))

    return poi


def filterFoodData(poi):
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

# def distance(city, stations):
#     lat1 = city['latitude']
#     lon1 = city['longitude']
#     lat2 = stations['latitude']
#     lon2 = stations['longitude']
#     R = 6371
#     p = np.pi / 180  # Pi/180
#     a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * \
#         np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
#     distance = (2 * R * np.arcsin(np.sqrt(a)))
#     return distance


def get_k_KM(data):
    # Code from https://towardsdatascience.com/k-means-clustering-using-pyspark-on-big-data-6214beacdc8b
    scale = StandardScaler(inputCol='features', outputCol='standardized')
    data_scale = scale.fit(data)
    data_scale = data_scale.transform(data)

    evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
                                    metricName='silhouette', distanceMeasure='squaredEuclidean')

    # Cost of k
    cost = np.zeros(20)
    for i in range(2, 20):
        gmm = KMeans(featuresCol='standardized', k=i)
        model = gmm.fit(data_scale)

        predictions = model.transform(data_scale)
        cost[i] = evaluator.evaluate(predictions)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, 20), cost[2:20])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    plt.show()


def cluster_KM(data, test_data, k):
    # Code from https://spark.apache.org/docs/latest/ml-clustering.html
    # Trains a k-means model.
    kmeans = KMeans(k=k)
    model = kmeans.fit(data)

    # Make predictions
    predictions = model.transform(data)
    test_predictions = model.transform(test_data)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    test_silhouette = evaluator.evaluate(test_predictions)
    print(
        f"KM silhouette with squared euclidean distance = {str(silhouette)} {str(test_silhouette)}")

    # Shows the result.
    # centers = model.clusterCenters()
    # print("Cluster Centers: ")
    # for center in centers:
    #     print(center)

    return predictions, test_predictions


def get_k_BKM(data):
    # Code from https://towardsdatascience.com/k-means-clustering-using-pyspark-on-big-data-6214beacdc8b
    scale = StandardScaler(inputCol='features', outputCol='standardized')
    data_scale = scale.fit(data)
    data_scale = data_scale.transform(data)

    evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
                                    metricName='silhouette', distanceMeasure='squaredEuclidean')

    # Cost of k
    cost = np.zeros(20)
    for i in range(2, 20):
        gmm = BisectingKMeans(featuresCol='standardized', k=i)
        model = gmm.fit(data_scale)

        predictions = model.transform(data_scale)
        cost[i] = evaluator.evaluate(predictions)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, 20), cost[2:20])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    plt.show()


def cluster_BKM(data, test_data, k):
    # Code from https://spark.apache.org/docs/latest/ml-clustering.html
    # Trains a bisecting k-means model.
    bkm = BisectingKMeans(k=k)
    model = bkm.fit(data)

    # Make predictions
    predictions = model.transform(data)
    test_predictions = model.transform(test_data)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    test_silhouette = evaluator.evaluate(test_predictions)
    print(
        f"BKM silhouette with squared euclidean distance = {str(silhouette)} {str(test_silhouette)}")

    # Shows the result.
    # print("Cluster Centers: ")
    # centers = model.clusterCenters()
    # for center in centers:
    #     print(center)

    return predictions, test_predictions


def get_k_GMM(data):
    # Code from https://towardsdatascience.com/k-means-clustering-using-pyspark-on-big-data-6214beacdc8b
    scale = StandardScaler(inputCol='features', outputCol='standardized')
    data_scale = scale.fit(data)
    data_scale = data_scale.transform(data)

    evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
                                    metricName='silhouette', distanceMeasure='squaredEuclidean')

    # Cost of k
    cost = np.zeros(20)
    for i in range(2, 20):
        gmm = GaussianMixture(featuresCol='standardized', k=i)
        model = gmm.fit(data_scale)

        predictions = model.transform(data_scale)
        cost[i] = evaluator.evaluate(predictions)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, 20), cost[2:20])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    plt.show()


def cluster_GMM(data, test_data, k):
    # Code from https://spark.apache.org/docs/latest/ml-clustering.html
    # Trains a bisecting gmm model.
    gmm = GaussianMixture(k=k)
    model = gmm.fit(data)

    # Make predictions
    predictions = model.transform(data)
    test_predictions = model.transform(test_data)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    test_silhouette = evaluator.evaluate(test_predictions)
    print(
        f"GMM silhouette with squared euclidean distance = {str(silhouette)} {str(test_silhouette)}")

    # print("Gaussians shown as a DataFrame: ")
    # model.gaussiansDF.show(truncate=False)

    return predictions, test_predictions


def ML(data, test_data):
    km_df, km_test = cluster_KM(data, test_data, 13)
    bkm_df, bkm_test = cluster_BKM(data, test_data, 14)
    gmm_df, gmm_test = cluster_GMM(data, test_data, 15)

    return km_df, km_test, bkm_df, bkm_test, gmm_df, gmm_test


def pandasDataframe(chain, nonchain):
    chain_pd = chain.toPandas()
    nonchain_pd = nonchain.toPandas()

    return chain_pd, nonchain_pd


def plotKMeansFig(chain, nonchain):
    color_labels = chain['prediction'].unique()
    rgb_values = sns.color_palette("husl", color_labels.size)
    color_map = dict(zip(color_labels, rgb_values))

    fig, (p1, p2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle('K-Means', fontsize=16)
    p1.scatter(chain.lat, chain.lon, c=chain.prediction.map(
        color_map), edgecolors=colors.to_rgba('black', 0.5))
    p1.set_title('Chain Food Locations', fontsize=14)
    p1.set_xlabel('Latitude', fontsize=14)
    p1.set_ylabel('Longitude', fontsize=14)
    p1.tick_params(labelsize=14)

    p2.scatter(nonchain.lat, nonchain.lon, c=nonchain.prediction.map(
        color_map), edgecolors=colors.to_rgba('black', 0.5))
    p2.set_title('Nonchain Food Locations', fontsize=14)
    p2.set_xlabel('Latitude', fontsize=14)
    p2.set_ylabel('Longitude', fontsize=14)
    p2.tick_params(labelsize=14)
    fig.savefig('figures/kmeans.png')


def plotBkmFig(chain, nonchain, output):
    color_labels = chain['prediction'].unique()
    rgb_values = sns.color_palette("husl", color_labels.size)
    color_map = dict(zip(color_labels, rgb_values))

    fig, (p1, p2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle('Bisecting KMeans', fontsize=16)
    p1.scatter(chain.lat, chain.lon, c=chain.prediction.map(
        color_map), edgecolors=colors.to_rgba('black', 0.5))
    p1.set_title('Chain Food Locations', fontsize=14)
    p1.set_xlabel('Latitude', fontsize=14)
    p1.set_ylabel('Longitude', fontsize=14)
    p1.tick_params(labelsize=14)

    p2.scatter(nonchain.lat, nonchain.lon, c=nonchain.prediction.map(
        color_map), edgecolors=colors.to_rgba('black', 0.5))
    p2.set_title('Nonchain Food Locations', fontsize=14)
    p2.set_xlabel('Latitude', fontsize=14)
    p2.set_ylabel('Longitude', fontsize=14)
    p2.tick_params(labelsize=14)
    fig.savefig(f'figures/{output}.png')


def plotGmmFig(chain, nonchain):
    color_labels = chain['prediction'].unique()
    rgb_values = sns.color_palette("husl", color_labels.size)
    color_map = dict(zip(color_labels, rgb_values))

    fig, (p1, p2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle('Gaussian Mixture Model', fontsize=16)
    p1.scatter(chain.lat, chain.lon, c=chain.prediction.map(
        color_map), edgecolors=colors.to_rgba('black', 0.5))
    p1.set_title('Chain Food Locations', fontsize=14)
    p1.set_xlabel('Latitude', fontsize=14)
    p1.set_ylabel('Longitude', fontsize=14)
    p1.tick_params(labelsize=14)

    p2.scatter(nonchain.lat, nonchain.lon, c=nonchain.prediction.map(
        color_map), edgecolors=colors.to_rgba('black', 0.5))
    p2.set_title('Nonchain Food Locations', fontsize=14)
    p2.set_xlabel('Latitude', fontsize=14)
    p2.set_ylabel('Longitude', fontsize=14)
    p2.tick_params(labelsize=14)
    fig.savefig('figures/gmm.png')


def main(inputs, output):
    poi = spark.read.json(inputs, schema=amenity_schema)
    gvData = filterGreaterVancouver(poi)
    chain, nonchain = filterFoodData(gvData)
    chain = chain.cache()
    nonchain = nonchain.cache()

    # Chain and non-chain food locations for each region
    bbyChain, bbyNonChain, vanChain, vanNonChain, richChain, \
        richNonChain, coqChain, coqNonChain, sryChain, sryNonChain, \
        mrChain, mrNonChain, deltaChain, deltaNonChain, lanChain, \
        lanNonChain, abbChain, abbNonChain = filterRegions(gvData)

    chain.write.json(output + '/chain', mode='overwrite', compression='gzip')
    nonchain.write.json(output + '/nonchain',
                        mode='overwrite', compression='gzip')

    chain_km, nonchain_km, chain_bkm, nonchain_bkm, chain_gmm, nonchain_gmm = ML(
        chain, nonchain)

    # get_k_BKM(bbyChain)
    # get_k_BKM(vanChain)
    # get_k_BKM(richChain)
    # get_k_BKM(coqChain)
    # get_k_BKM(sryChain)
    # get_k_BKM(mrChain)
    # get_k_BKM(deltaChain)
    # get_k_BKM(lanChain)
    # get_k_BKM(abbChain)

    # chain.show()
    # chain_km.show()

    print("\nBKM silhouette scores for each region")
    chain_bkm_bby, nonchain_bkm_bby = cluster_BKM(bbyChain, bbyNonChain, 4)
    chain_bkm_van, nonchain_bkm_van = cluster_BKM(vanChain, vanNonChain, 4)
    chain_bkm_rich, nonchain_bkm_rich = cluster_BKM(richChain, richNonChain, 8)
    chain_bkm_coq, nonchain_bkm_coq = cluster_BKM(coqChain, coqNonChain, 10)
    chain_bkm_sry, nonchain_bkm_sry = cluster_BKM(sryChain, sryNonChain, 9)
    chain_bkm_mr, nonchain_bkm_mr = cluster_BKM(mrChain, mrNonChain, 3)
    chain_bkm_delta, nonchain_bkm_delta = cluster_BKM(
        deltaChain, deltaNonChain, 4)
    chain_bkm_lan, nonchain_bkm_lan = cluster_BKM(lanChain, lanNonChain, 6)
    chain_bkm_abb, nonchain_bkm_abb = cluster_BKM(abbChain, abbNonChain, 4)

    totalCountChain = chain.count()
    totalCountNonChain = nonchain.count()
    bbyCountChain = bbyChain.count()
    bbyCountNonChain = bbyNonChain.count()
    vanCountChain = vanChain.count()
    vanCountNonChain = vanNonChain.count()
    richCountChain = richChain.count()
    richCountNonChain = richNonChain.count()
    coqCountChain = coqChain.count()
    coqCountNonChain = coqNonChain.count()
    sryCountChain = sryChain.count()
    sryCountNonChain = sryNonChain.count()
    mrCountChain = mrChain.count()
    mrCountNonChain = mrNonChain.count()
    deltaCountChain = deltaChain.count()
    deltaCountNonChain = deltaNonChain.count()
    lanCountChain = lanChain.count()
    lanCountNonChain = lanNonChain.count()
    abbCountChain = abbChain.count()
    abbCountNonChain = abbNonChain.count()

    counts = np.array([[totalCountChain, totalCountNonChain],
                      [bbyCountChain, bbyCountNonChain],
                      [vanCountChain, vanCountNonChain],
                      [richCountChain, richCountNonChain],
                      [coqCountChain, coqCountNonChain],
                      [sryCountChain, sryCountNonChain],
                      [mrCountChain, mrCountNonChain],
                      [deltaCountChain, deltaCountNonChain],
                      [lanCountChain, lanCountNonChain],
                      [abbCountChain, abbCountNonChain]])

    regions = ["Greater Vancouver", "Burnaby", "Vancouver", "Richmond",
               "Coquitlam", "Surrey", "Maple Ridge", "Delta", "Langley", "Abbotsford"]

    restaurantCounts = pd.DataFrame(
        counts, columns=["Chain", "NonChain"], index=regions)
    counts_plt = restaurantCounts.plot.bar()
    counts_plt.set_ylabel('Count', fontsize=14)
    counts_plt.tick_params(labelsize=14)
    counts_plt.xaxis.set_tick_params(rotation=25)
    counts_plt.legend(fontsize=14)
    counts_fig = counts_plt.get_figure()
    counts_fig.set_size_inches(16, 9)
    counts_fig.savefig('figures/counts.png')

    # Convert pypark dataframes to panda dataframe for plotting
    # chain_km_pd, nonchain_km_pd = pandasDataframe(chain_km, nonchain_km)
    chain_bkm_pd, nonchain_bkm_pd = pandasDataframe(chain_bkm, nonchain_bkm)
    chain_bby_pd, nonchain_bby_pd = pandasDataframe(
        chain_bkm_bby, nonchain_bkm_bby)
    chain_van_pd, nonchain_van_pd = pandasDataframe(
        chain_bkm_van, nonchain_bkm_van)
    chain_rich_pd, nonchain_rich_pd = pandasDataframe(
        chain_bkm_rich, nonchain_bkm_rich)
    chain_coq_pd, nonchain_coq_pd = pandasDataframe(
        chain_bkm_coq, nonchain_bkm_coq)
    chain_sry_pd, nonchain_sry_pd = pandasDataframe(
        chain_bkm_sry, nonchain_bkm_sry)
    chain_mr_pd, nonchain_mr_pd = pandasDataframe(
        chain_bkm_mr, nonchain_bkm_mr)
    chain_delta_pd, nonchain_delta_pd = pandasDataframe(
        chain_bkm_delta, nonchain_bkm_delta)
    chain_lan_pd, nonchain_lan_pd = pandasDataframe(
        chain_bkm_lan, nonchain_bkm_lan)
    chain_abb_pd, nonchain_abb_pd = pandasDataframe(
        chain_bkm_abb, nonchain_bkm_abb)
    # chain_gmm_pd, nonchain_gmm_pd = pandasDataframe(chain_gmm, nonchain_gmm)

    # Plot the results
    # # KMeans
    # plotKMeansFig(chain_km_pd, nonchain_km_pd)

    # BKM
    plotBkmFig(chain_bkm_pd, nonchain_bkm_pd, 'bkm-all')
    plotBkmFig(chain_bby_pd, nonchain_bby_pd, 'bkm-bby')
    plotBkmFig(chain_van_pd, nonchain_van_pd, 'bkm-van')
    plotBkmFig(chain_rich_pd, nonchain_rich_pd, 'bkm-rich')
    plotBkmFig(chain_coq_pd, nonchain_coq_pd, 'bkm-coq')
    plotBkmFig(chain_sry_pd, nonchain_sry_pd, 'bkm-sry')
    plotBkmFig(chain_mr_pd, nonchain_mr_pd, 'bkm-mr')
    plotBkmFig(chain_delta_pd, nonchain_delta_pd, 'bkm-delta')
    plotBkmFig(chain_lan_pd, nonchain_lan_pd, 'bkm-lan')
    plotBkmFig(chain_abb_pd, nonchain_abb_pd, 'bkm-abb')

    # # GMM
    # plotGmmFig(chain_gmm_pd, nonchain_gmm_pd)

    # poi.show()
    poi = poi.coalesce(1)  # ~1MB after the filtering
    poi.write.json(output + '/data', mode='overwrite', compression='gzip')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
