# Building and Visualizing Models

## Introduction 

### Oracle AutoML

The Oracle AutoML solution automatically tunes a model class to produce the best models. It works with any supervised prediction task, such as classification or regression. It supports binary and multi-class classifications, as well as regression problems. `AutoML` automates three major stages of the ML pipeline, feature selection, algorithm selection, and hyperparameter tuning. These pieces are combined into a pipeline which automatically optimizes the whole process with minimal user interaction.

Oracle AutoML uses the `OracleAutoMLProvider` object to delegates the model training to AutoML.

In addition, AutoML has a pipeline-level Python API that quickly jump-starts the data science process with a quality tuned model. It selects the appropriate features and model class for a given prediction task.

### Objectives

In this lab, you will:
* Featuring Engineering
* Training and Hypertuning with AutoML

### Prerequisites

This lab assumes you have:
* An Oracle account
* All previous labs successfully completed

## Featuring Engineering : Label Encoding Categorical Features

```python
from ads.dataset.label_encoder import DataFrameLabelEncoder
ds_encoded = DataFrameLabelEncoder().fit_transform(transformed_ds.to_pandas_dataframe())

ds_encoded.head()
```

## Training and Hypertuning Logistic Regression and LGBM Classifiers with Oracle AutoML

```python
try:
    if train is not None:
        del train
    if test is not None:
        del test
except NameError:
    pass

train_original, test_original = transformed_ds.drop_columns(['late_mort_rent_pmts','country']).train_test_split()
```

```python
train, test = DatasetFactory.from_dataframe(ds_encoded, target='target').drop_columns(['late_mort_rent_pmts','country']).train_test_split()
automl = AutoML(train, provider=OracleAutoMLProvider())
model, baseline = automl.train(model_list=['LogisticRegression', 'LGBMClassifier'],  random_state = 42, 
                               score_metric = "roc_auc", time_budget=160)
```

#### LGBM 'Best' Model Based on Optimized Roc_AUC

```python
model.show_in_notebook()
```

#### Hypertuned Parameters Resulting in a ~88% Accuracy in Predicting if a Person will be Delinquent on their Rent

```python
from sklearn.metrics import get_scorer
accuracy_scorer = get_scorer("accuracy") # Works with any sklearn scoring function.

print("Oracle AutoML accuracy on test data:", 
      model.score(test.X, test.y, score_fn = accuracy_scorer))
```

```python
model.selected_model_params_
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