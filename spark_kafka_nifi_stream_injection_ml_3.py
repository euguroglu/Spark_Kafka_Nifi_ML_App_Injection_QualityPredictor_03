import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, VectorSizeHint, StandardScaler, MinMaxScaler
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix


# Create spark session
spark = SparkSession\
    .builder\
    .master('local[2]')\
    .appName('injection_predictor')\
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:2.4.1')\
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .getOrCreate()

"""
Machine learning part
"""
# We are currently not training model on the fly
# Instead we are loading saved model directly
# To see machine learning training check "spark_kafka_nifi_stream_injection_ml.py file"
# Loading the model
model = PipelineModel.load('model')

"""
Streaming Part
"""
#Read from kafka topic "injection"
kafka_df = spark \
    .readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', "localhost:9092") \
    .option("startingOffsets", "earliest") \
    .option('subscribe', 'injection') \
    .option("failOnDataLoss", "false") \
    .load()

#Define schema
schema = StructType([StructField('reason', DoubleType()),\
                        StructField('tmpMoldZone25', DoubleType()),\
                        StructField('timCool1', DoubleType()),\
                        StructField('tmpBarrel2Zone3', DoubleType()),\
                        StructField('tmpMoldZone3', DoubleType()),\
                        StructField('tmpBarrel2Zone4', DoubleType()),\
                        StructField('tmpFlange1', DoubleType()),\
                        StructField('tmpMoldZone4', DoubleType()),\
                        StructField('tmpBarrel2Zone1', DoubleType()),\
                        StructField('tmpFlange2', DoubleType()),\
                        StructField('tmpMoldZone1', DoubleType()),\
                        StructField('volCushion1', DoubleType()),\
                        StructField('tmpBarrel2Zone2', DoubleType()),\
                        StructField('tmpMoldZone2', DoubleType()),\
                        StructField('volCushion2', DoubleType()),\
                        StructField('prsBackSpec2', DoubleType()),\
                        StructField('prsBackSpec1', DoubleType()),\
                        StructField('tmpMoldZone9', DoubleType()),\
                        StructField('spdInjection2', DoubleType()),\
                        StructField('tmpMoldZone7', DoubleType()),\
                        StructField('tmpMoldZone8', DoubleType()),\
                        StructField('tmpOil', DoubleType()),\
                        StructField('tmpMoldZone5', DoubleType()),\
                        StructField('tmpMoldZone6', DoubleType()),\
                        StructField('tmpMoldZone19', DoubleType()),\
                        StructField('tmpMoldZone18', DoubleType()),\
                        StructField('volTransfer2', DoubleType()),\
                        StructField('tmpMoldZone15', DoubleType()),\
                        StructField('volTransfer1', DoubleType()),\
                        StructField('tmpMoldZone14', DoubleType()),\
                        StructField('tmpMoldZone17', DoubleType()),\
                        StructField('tmpMoldZone16', DoubleType()),\
                        StructField('timTransfer2', DoubleType()),\
                        StructField('timTransfer1', DoubleType()),\
                        StructField('velPlasticisation2', DoubleType()),\
                        StructField('velPlasticisation1', DoubleType()),\
                        StructField('timMoldClose', DoubleType()),\
                        StructField('tmpBarrel1Zone5', DoubleType()),\
                        StructField('tmpMoldZone22', DoubleType()),\
                        StructField('tmpBarrel1Zone4', DoubleType()),\
                        StructField('tmpMoldZone21', DoubleType()),\
                        StructField('tmpMoldZone24', DoubleType()),\
                        StructField('tmpBarrel1Zone6', DoubleType()),\
                        StructField('tmpMoldZone23', DoubleType()),\
                        StructField('prsPomp1', DoubleType()),\
                        StructField('tmpBarrel1Zone1', DoubleType()),\
                        StructField('prsPomp2', DoubleType()),\
                        StructField('tmpBarrel1Zone3', DoubleType()),\
                        StructField('tmpMoldZone20', DoubleType()),\
                        StructField('tmpBarrel1Zone2', DoubleType()),\
                        StructField('volShot1', DoubleType()),\
                        StructField('volPlasticisation2', DoubleType()),\
                        StructField('volShot2', DoubleType()),\
                        StructField('volPlasticisation1', DoubleType()),\
                        StructField('timFill1', DoubleType()),\
                        StructField('timFill2', DoubleType()),\
                        StructField('timMoldOpen', DoubleType()),\
                        StructField('tmpMoldZone11', DoubleType()),\
                        StructField('tmpMoldZone10', DoubleType()),\
                        StructField('tmpMoldZone13', DoubleType()),\
                        StructField('tmpMoldZone12', DoubleType()),\
                        StructField('prsHoldSpec2', DoubleType()),\
                        StructField('tmpNozle2', DoubleType()),\
                        StructField('prsHoldSpec1', DoubleType()),\
                        StructField('tmpNozle1', DoubleType()),\
                        StructField('prsTransferSpec2', DoubleType()),\
                        StructField('prsTransferSpec1', DoubleType()),\
                        StructField('prsInjectionSpec1', DoubleType()),\
                        StructField('prsInjectionSpec2', DoubleType()),\
                        StructField('timCycle', DoubleType()),\
                        StructField('frcClamp', DoubleType()),\
                        StructField('timPlasticisation1', DoubleType()),\
                        StructField('timPlasticisation2', DoubleType())])

#Print schema to review
kafka_df.printSchema()

#Deserialize json object and apply schema
value_df = kafka_df.select(from_json(col("value").cast("string"),schema).alias("value"))

explode_df = value_df.selectExpr("value.reason","value.tmpMoldZone25", "value.timCool1", \
                                "value.tmpBarrel2Zone3","value.tmpMoldZone3", "value.tmpBarrel2Zone4", \
                                "value.tmpFlange1","value.tmpMoldZone4", "value.tmpBarrel2Zone1", \
                                "value.tmpFlange2","value.tmpMoldZone1", "value.volCushion1", \
                                "value.tmpBarrel2Zone2","value.tmpMoldZone2", "value.volCushion2", \
                                "value.prsBackSpec2","value.prsBackSpec1", "value.tmpMoldZone9", \
                                "value.spdInjection2","value.tmpMoldZone7", "value.tmpMoldZone8", \
                                "value.tmpOil","value.tmpMoldZone5", "value.tmpMoldZone6", \
                                "value.tmpMoldZone19","value.tmpMoldZone18", "value.volTransfer2", \
                                "value.tmpMoldZone15","value.volTransfer1", "value.tmpMoldZone14", \
                                "value.tmpMoldZone17","value.tmpMoldZone16", "value.timTransfer2", \
                                "value.timTransfer1","value.velPlasticisation2", "value.velPlasticisation1", \
                                "value.timMoldClose","value.tmpBarrel1Zone5", "value.tmpMoldZone22", \
                                "value.tmpBarrel1Zone4","value.tmpMoldZone21", "value.tmpMoldZone24", \
                                "value.tmpBarrel1Zone6","value.tmpMoldZone23", "value.prsPomp1", \
                                "value.tmpBarrel1Zone1","value.prsPomp2", "value.tmpBarrel1Zone3", \
                                "value.tmpMoldZone20","value.tmpBarrel1Zone2", "value.volShot1", \
                                "value.volPlasticisation2","value.volShot2", "value.volPlasticisation1", \
                                "value.timFill1","value.timFill2", "value.timMoldOpen", \
                                "value.tmpMoldZone11","value.tmpMoldZone10", "value.tmpMoldZone13", \
                                "value.tmpMoldZone12","value.prsHoldSpec2", "value.tmpNozle2", \
                                "value.prsHoldSpec1","value.tmpNozle1", "value.prsTransferSpec2", \
                                "value.prsTransferSpec1","value.prsInjectionSpec1", "value.prsInjectionSpec2", \
                                "value.timCycle","value.frcClamp", "value.timPlasticisation1","value.timPlasticisation2")
#Print schema to review
explode_df = explode_df.drop('reason')
explode_df.printSchema()

pred_results_stream = model.transform(explode_df)
#Remove feature column
pred_results_stream_simplified = pred_results_stream.selectExpr("timCycle", "prediction")

# kafka_df = pred_results_stream_simplified.select("*")
# kafka_df = kafka_df.selectExpr("cast(timCycle as string) timCycle", "prediction")
#
# kafka_target_df = kafka_df.selectExpr("timCycle as key",
#                                              "to_json(struct(*)) as value")
#
# kafka_target_df.printSchema()
#
# nifi_query = kafka_target_df \
#         .writeStream \
#         .queryName("Notification Writer") \
#         .format("kafka") \
#         .option("kafka.bootstrap.servers", "localhost:9092") \
#         .option("topic", "injection2") \
#         .outputMode("append") \
#         .option("checkpointLocation", "chk-point-dir") \
#         .start()
#
# nifi_query.awaitTermination()
# ## Below command used to preview results on the console before inserting data to database
# #Sink result to console
window_query = pred_results_stream_simplified.writeStream \
     .format("console") \
     .outputMode("append") \
     .trigger(processingTime="10 seconds") \
     .start()

window_query.awaitTermination()
