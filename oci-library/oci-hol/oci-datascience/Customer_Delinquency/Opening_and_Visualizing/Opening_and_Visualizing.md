# Opening and Visualizing Datasets using `ADS`

## Introduction

*fill in ...

Estimated Lab Time: -- 10 minutes

### Objectives

In this lab, you will:
* The first step is to load in the dataset
* Visualizing the Dataset using Oracle ADS SDK
* Using Oracle ADS SKD get_recomendations

### Prerequisites

This lab assumes you have:
* An Oracle account
* All previous labs successfully completed

## Binary Classification

Binary classification is a technique of classifying observations into one of two groups. In this notebook, the two groups are those mortgage owners or rental tenants that will be delinquent. 

Given the features in the data, the model determines the optimal criteria for classifying an observation as leaving or not leaving. This optimization is based on the training data. However, some of the data to test the model's preformance is reserved. Models can overfit on the training data, that is learn the noise in a dataset, and then it won't do a good job at predicting the results on new data (test data). Since you already know the truth for the data in the training dataset, you are really interested in how well it performs on the test data.

## The Dataset

This is a fictional data set and contains 1k rows. There are 17 features with 9 ordinal features, 7 categorical features, and 1 continous feature. The features include basic demographic information, income level, credit balance and other attributes associated with the customer's financial background.

The first step is to load in the dataset. To do this the `DatasetFactory` singleton object is used. It is part of the `ADS` library and is a powerful class to work with datasets from different sources.


```python
with cx_Oracle.connect(os.environ["ADW_USER"], os.environ["ADW_PASSWORD"], os.environ["ADW_SID"]) as ora_conn:
  df = pd.read_sql('''
   select * from beta.cust_agg order by cust_id  ''', con=ora_conn)

df.columns = map(str.lower, df.columns)
ds = DatasetFactory.from_dataframe(df, target='target')
```

HBox(children=(HTML(value='loop1'), FloatProgress(value=0.0, max=4.0), HTML(value='')))

```python
ds.shape
```

(1000, 17)

```python
ds.head()
```

## Visualizing the Dataset using Oracle ADS SDK

The `show_in_notebook()` method can be applied to the dataset itself. When this is done the following is produced:

  - Summary, which shows a brief description of the dataset, shape, and some break down by feature type.
  - Features, the visualization created on a dataset sample to give an idea of distribution for each feature.
  - Correlations, which shows how every feature (numeric and categorical) are correlated.
  - Warnings, about the condition of your data including near zero variance features, high cardinality features, features with missing values, and features with a lot of zero values.

```python
ds.show_in_notebook()
```

#### Using ADS show_corr() Method for Automatical Correlation Plots

```python
ds.show_corr()
```

### Using Oracle ADS SKD get_recomendations

#### Here we use the get_recommendations() method to help drive the data cleansing step. Alternatively, we can use the auto_transform() to automatically apply all the recommended transformations to our data set

```python
ds.get_recommendations()
```

Output()

```python
transformed_ds = ds.auto_transform(fix_imbalance=True)
transformed_ds.visualize_transforms()
```

HBox(children=(HTML(value='loop1'), FloatProgress(value=0.0, max=7.0), HTML(value='')))

```python
transformed_ds.head()
```

## Learn More

### OCI Data Science - Useful Tips
<details>
<summary><font size="3">Check for Public Internet Access</font></summary>

```python
import requests
response = requests.get("https://oracle.com")
assert response.status_code==200, "Internet connection failed"
```
</details>
<details>
<summary><font size="3">Helpful Documentation </font></summary>
<ul><li><a href="https://docs.cloud.oracle.com/en-us/iaas/data-science/using/data-science.htm">Data Science Service Documentation</a></li>
<li><a href="https://docs.cloud.oracle.com/iaas/tools/ads-sdk/latest/index.html">ADS documentation</a></li>
</ul>
</details>
<details>
<summary><font size="3">Typical Cell Imports and Settings for ADS</font></summary>

```python
%load_ext autoreload
%autoreload 2
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

import ads
from ads.dataset.factory import DatasetFactory
from ads.automl.provider import OracleAutoMLProvider
from ads.automl.driver import AutoML
from ads.evaluations.evaluator import ADSEvaluator
from ads.common.data import ADSData
from ads.explanations.explainer import ADSExplainer
from ads.explanations.mlx_global_explainer import MLXGlobalExplainer
from ads.explanations.mlx_local_explainer import MLXLocalExplainer
from ads.catalog.model import ModelCatalog
from ads.common.model_artifact import ModelArtifact
```
</details>
<details>
<summary><font size="3">Useful Environment Variables</font></summary>

```python
import os
print(os.environ["NB_SESSION_COMPARTMENT_OCID"])
print(os.environ["PROJECT_OCID"])
print(os.environ["USER_OCID"])
print(os.environ["TENANCY_OCID"])
print(os.environ["NB_REGION"])
```
</details>

## Acknowledgements
* **Author** - Jeremy Mendez, Senior Cloud Engineer
* **Contributors** - Aaron Whitman, Associate Cloud Engineer 
* **Last Updated Date** - October, 2021