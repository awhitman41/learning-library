# Building and Visualizing Models

## Introduction 

### Oracle AutoML

The Oracle AutoML solution automatically tunes a model class to produce the best models. It works with any supervised prediction task, such as classification or regression. It supports binary and multi-class classifications, as well as regression problems. `AutoML` automates three major stages of the ML pipeline, feature selection, algorithm selection, and hyperparameter tuning. These pieces are combined into a pipeline which automatically optimizes the whole process with minimal user interaction.

Oracle AutoML uses the `OracleAutoMLProvider` object to delegates the model training to AutoML.

In addition, AutoML has a pipeline-level Python API that quickly jump-starts the data science process with a quality tuned model. It selects the appropriate features and model class for a given prediction task.

![image.png](attachment:image.png)

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

HBox(children=(HTML(value='loop1'), FloatProgress(value=0.0, max=4.0), HTML(value='')))

#### LGBM 'Best' Model Based on Optimized Roc_AUC

```python
model.show_in_notebook()
```

[
	{
		"metadata": {
			"outputType": "display_data",
			"metadata": {}
		},
		"outputItems": [
			{
				"mimeType": "text/html",
				"data": "<table border=\"1\" class=\"dataframe\">\n  <tbody>\n    <tr>\n      <td>Model Name</td>\n      <td>AutoML Classifier</td>\n    </tr>\n    <tr>\n      <td>Target Variable</td>\n      <td>target</td>\n    </tr>\n    <tr>\n      <td>Selected Algorithm</td>\n      <td>LGBMClassifier</td>\n    </tr>\n    <tr>\n      <td>Task</td>\n      <td>classification</td>\n    </tr>\n    <tr>\n      <td>Training Dataset Size</td>\n      <td>(890, 12)</td>\n    </tr>\n    <tr>\n      <td>CV</td>\n      <td>5</td>\n    </tr>\n    <tr>\n      <td>Optimization Metric</td>\n      <td>roc_auc</td>\n    </tr>\n    <tr>\n      <td>Selected Hyperparameters</td>\n      <td>{'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 101, 'n_jobs': 2, 'num_leaves': 31, 'objective': None, 'random_state': 42, 'reg_alpha': 0, 'reg_lambda': 1, 'silent': True, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0}</td>\n    </tr>\n    <tr>\n      <td>Initial Number of Features</td>\n      <td>12</td>\n    </tr>\n    <tr>\n      <td>Initial Features</td>\n      <td>Index(['age', 'yrs_current_employer', 'marital_status', 'rent_own',\n       'mortgage_amt', 'job_type', 'insuff_funds_incidents', 'income_level',\n       'education', 'credit_balance', 'transaction_cnt', 'avg_transaction'],\n      dtype='object')</td>\n    </tr>\n    <tr>\n      <td>Selected Number of Features</td>\n      <td>10</td>\n    </tr>\n    <tr>\n      <td>Selected Features</td>\n      <td>[age, yrs_current_employer, marital_status, mortgage_amt, insuff_funds_incidents, income_level, education, credit_balance, transaction_cnt, avg_transaction]</td>\n    </tr>\n  </tbody>\n</table>"
			},
			{
				"mimeType": "text/plain",
				"data": "<IPython.core.display.HTML object>"
			}
		]
	},
	{
		"metadata": {
			"outputType": "execute_result",
			"executionCount": 20,
			"metadata": {}
		},
		"outputItems": [
			{
				"mimeType": "text/plain",
				"data": "[['Model Name', 'AutoML Classifier'],\n ['Target Variable', 'target'],\n ['Selected Algorithm', 'LGBMClassifier'],\n ['Task', 'classification'],\n ['Training Dataset Size', (890, 12)],\n ['CV', 5],\n ['Optimization Metric', 'roc_auc'],\n ['Selected Hyperparameters',\n  {'boosting_type': 'gbdt',\n   'class_weight': None,\n   'colsample_bytree': 1.0,\n   'importance_type': 'split',\n   'learning_rate': 0.1,\n   'max_depth': -1,\n   'min_child_samples': 20,\n   'min_child_weight': 0.001,\n   'min_split_gain': 0.0,\n   'n_estimators': 101,\n   'n_jobs': 2,\n   'num_leaves': 31,\n   'objective': None,\n   'random_state': 42,\n   'reg_alpha': 0,\n   'reg_lambda': 1,\n   'silent': True,\n   'subsample': 1.0,\n   'subsample_for_bin': 200000,\n   'subsample_freq': 0}],\n ['Initial Number of Features', 12],\n ['Initial Features',\n  Index(['age', 'yrs_current_employer', 'marital_status', 'rent_own',\n         'mortgage_amt', 'job_type', 'insuff_funds_incidents', 'income_level',\n         'education', 'credit_balance', 'transaction_cnt', 'avg_transaction'],\n        dtype='object')],\n ['Selected Number of Features', 10],\n ['Selected Features',\n  ['age',\n   'yrs_current_employer',\n   'marital_status',\n   'mortgage_amt',\n   'insuff_funds_incidents',\n   'income_level',\n   'education',\n   'credit_balance',\n   'transaction_cnt',\n   'avg_transaction']]]"
			}
		]
	}
]

*Ask Jeremy*

[['Model Name', 'AutoML Classifier'],
 ['Target Variable', 'target'],
 ['Selected Algorithm', 'LGBMClassifier'],
 ['Task', 'classification'],
 ['Training Dataset Size', (890, 12)],
 ['CV', 5],
 ['Optimization Metric', 'roc_auc'],
 ['Selected Hyperparameters',
  {'boosting_type': 'gbdt',
   'class_weight': None,
   'colsample_bytree': 1.0,
   'importance_type': 'split',
   'learning_rate': 0.1,
   'max_depth': -1,
   'min_child_samples': 20,
   'min_child_weight': 0.001,
   'min_split_gain': 0.0,
   'n_estimators': 101,
   'n_jobs': 2,
   'num_leaves': 31,
   'objective': None,
   'random_state': 42,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'silent': True,
show more (open the raw output data in a text editor) ...

   'insuff_funds_incidents',
   'income_level',
   'education',
   'credit_balance',
   'transaction_cnt',
   'avg_transaction']]]

#### Hypertuned Parameters Resulting in a ~88% Accuracy in Predicting if a Person will be Delinquent on their Rent

```python
from sklearn.metrics import get_scorer
accuracy_scorer = get_scorer("accuracy") # Works with any sklearn scoring function.

print("Oracle AutoML accuracy on test data:", 
      model.score(test.X, test.y, score_fn = accuracy_scorer))
```

Oracle AutoML accuracy on test data: 0.9090909090909091

```python
model.selected_model_params_
```

{'boosting_type': 'gbdt',
 'class_weight': None,
 'colsample_bytree': 1.0,
 'importance_type': 'split',
 'learning_rate': 0.1,
 'max_depth': -1,
 'min_child_samples': 20,
 'min_child_weight': 0.001,
 'min_split_gain': 0.0,
 'n_estimators': 101,
 'n_jobs': 2,
 'num_leaves': 31,
 'objective': None,
 'random_state': 42,
 'reg_alpha': 0,
 'reg_lambda': 1,
 'silent': True,
 'subsample': 1.0,
 'subsample_for_bin': 200000,
 'subsample_freq': 0}

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