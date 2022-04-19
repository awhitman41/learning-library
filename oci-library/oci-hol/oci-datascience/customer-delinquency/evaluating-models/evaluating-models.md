# Evaluating Models using `ADSEvaluator`

## Introduction

One of the key advantages of `ADS` is the ability to quickly evaluate any models. `ADS` supports evaluating:

- Regression
- Binary classification
- Multiclass classification

`ADS` allows you to provide your own evaluation function (given `y_true` and `y_pred` series) for any esoteric calculation that you want to run.

### Objectives

In this lab, you will:
* Provide your own evaluation function

### Prerequisites

This lab assumes you have:
* An Oracle account
* All previous labs successfully completed

## Provide your own evaluation function

Next, you examine the plots that are commonly used to evaluate model performance. These include the precision-recall, ROC, lift, and gain plots. Each model under study is plotted together. This allows for easy comparison. In addition, the normalized confusion matrices are provided.

```python
valuator = ADSEvaluator(test, models=[model, baseline], 
                         training_data=train,
                         show_full_name=True,
                         positive_class=True)
evaluator.show_in_notebook()
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