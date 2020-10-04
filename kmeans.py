#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Creamos la sesion de spark
if __name__=="__main__":
    try:
        from pyspark.sql import SparkSession
    except:
        import findspark
        findspark.init()
        from pyspark.sql import SparkSession
    spark=SparkSession.builder     .appName("kmeans")     .getOrCreate()


# In[2]:


# Cargamos los datos usando spark
data = spark.read.csv("s3://bucketalfredo.s3.amazonaws.com/dataframes/titanic.csv",header=True,inferSchema=True)
data.show(10)


# In[3]:


from pyspark.ml.feature import StringIndexer,OneHotEncoderEstimator,VectorAssembler,MinMaxScaler


# In[4]:


sex_enc=StringIndexer(inputCol='sex',outputCol='sex_enc',stringOrderType='alphabetAsc')
embark_enc=StringIndexer(inputCol='embark_town',outputCol='embark_enc',stringOrderType='alphabetAsc')
class_enc=StringIndexer(inputCol='class',outputCol='class_enc',stringOrderType='alphabetAsc')
alone_enc=StringIndexer(inputCol='alone',outputCol='alone_enc',stringOrderType='alphabetAsc')


# In[5]:


ohe=OneHotEncoderEstimator(inputCols=['sex_enc','embark_enc'],outputCols=['sex_ohe','embark_ohe'])


# In[6]:


vec=VectorAssembler(inputCols=['age','n_siblings_spouses','parch','fare','class_enc','alone_enc',
                               'sex_ohe','embark_ohe'],outputCol='features')


# In[7]:


sca=MinMaxScaler(inputCol='features',outputCol='features_mm')


# In[9]:


from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# In[10]:


clu=KMeans(k=3)
pipe=Pipeline(stages=[sex_enc,embark_enc,class_enc,alone_enc,ohe,vec,sca,clu])


# In[12]:


modelo=pipe.fit(data)


# In[14]:


prediccion=modelo.transform(data)
#prediccion.select('prediction').show(5)


# In[15]:


evaluacion=ClusteringEvaluator()
print('Silhoutte', evaluacion.evaluate(prediccion))

