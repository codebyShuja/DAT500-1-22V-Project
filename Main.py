import time

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, StopWordsRemover
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
import numpy as np 
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext
from nltk.stem.snowball import SnowballStemmer
import pyspark.sql.functions as f

#start = time.time()

spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()
sc=spark.sparkContextspark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()
sc=spark.sparkContext
sqlContext = SQLContext(sc)
data1 = spark.read.json('kindle.json')
data2 = spark.read.json('meta.json')
data1 = data1.unionByName(data2,allowMissingColumns=True)
data1.head()
print("Data Schema")
data1.printSchema()
data1 = data1.drop("category","tech1","description","rank","title","fit","also_buy","tech2","brand","feature","also_view","details","main_cat","similar_item","date","price","imageURL","imageURLHighRes","image","reviewTime","reviewerName","style","summary","unixReviewTime","verified","vote")
print("Data Schema After Drop")
data1.printSchema()
data1.show()
data1.select('reviewText').show(1)
df_clean = data1.select('asin', (lower(regexp_replace('reviewText', "[^a-zA-Z\\s]", "")).alias('text')))
print("Remove Special Characters")
df_clean.show(10)
tokenizer = Tokenizer(inputCol='text', outputCol='words_token')
df_words_token = tokenizer.transform(df_clean).select('asin', 'words_token')
print("Tokenizer Function")
df_words_token.show(10)
remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
df_words_no_stopw = remover.transform(df_words_token).select('asin', 'words_clean')
print("Stop words Remove Function")
df_words_no_stopw.show(10)


analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score
def change_polarity(pol_val):
    if pol_val >0:
        return "positive"
    elif pol_val < 0:
        return "negative"
    else:
        return "neutral"
def sentement_checker(text):
    sent = TextBlob(text).sentiment.polarity
    return sent

#end = time.time()
#total_time = end - start
#print("TOTAL TIME"+ str(total_time))

print("Sentement Check")
checked_sentement = udf(lambda x: sentement_checker(x), DoubleType())
data1 = data1.withColumn('Sentements', checked_sentement('reviewText'))
print("Polarity Check")
checked_polarity = udf(lambda x: change_polarity(x), StringType())
data1 = data1.withColumn('Polarity', checked_polarity('Sentements'))

print("Vader Score Check ")
sentiment_analyzer_scores_udf = udf(lambda x: sentiment_analyzer_scores(x), StringType())
data1 = data1.withColumn('vader_score', sentiment_analyzer_scores_udf('reviewText'))
data1.show()
def generate_udf(constant_var="Correct"):
    def test(col1, col2):
        if col1 == "positive" and  int(col2) >= 5:
            return constant_var
        elif col1 == "negative" and  int(col2) < 5:
            return constant_var
        elif col1 == "neutral":
            return constant_var
        else:
            return "Incorrect"
    return f.udf(test, StringType())
def acc_pre(pol, rate):
    if rate >= 5 and pol == "positive":
        return "Correct"
    elif rate < 5 and pol == "negative":
        return "Correct"
    elif pol == "neutral":
        return"Correct"
    else:
        return "Incorrect"
data1 = data1.withColumn('Results', generate_udf('Correct')(f.col('Polarity'), f.col('overall')))
data1.select(col("reviewText"), col("Sentements"),col("Polarity")).show()
data1.select(col("vader_score")).show(truncate=False)

temp_df = data1.groupBy("overall").count()
temp_df.sort(col("Count").desc()).show()

data1 = data1.withColumn("target", lit(0))
df_ML = data1.select(col("reviewText"), col("target"))
df_ML.dropna()
df_ML.printSchema()
df_ML.show()
(train_set, val_set, test_set) = df_ML.randomSplit([0.98, 0.01, 0.01], seed = 2000)
tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
lr = LogisticRegression(maxIter=100)
pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx, lr])
pipelineFit = pipeline.fit(train_set)
predictions = pipelineFit.transform(val_set)
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())
print("Accuracy is: ", accuracy)