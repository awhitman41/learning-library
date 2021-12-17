# Introduction

## Overview

In this workshop, an employee attrition dataset is used. You start by doing an exploratory data analysis (EDA) to understand the data. Then a model is trained using `AutoML`. The model is used to make predictions and evaluate the model to determine how well it generalizes to new data. Then you use machine learning explainability (MLX) to understand the global and local model behavior. You do all of this using the Oracle Accelerated Data Science (`ADS`) library.

**Important:**

Placeholder text for required values are surrounded by angle brackets that must be removed when adding the indicated content. For example, when adding a database name to `database_name = "<database_name>"` would become `database_name = "production"`. 

### **Business Use Case**

Organizations can face significant costs resulting from employee turnover. Some costs are tangible, such as training expenses and the time it takes from when an employee starts to when they become a productive team member. Generally, the most important costs are intangible. Consider what is lost when a productive employee quits, such as corporate knowledge, new product ideas, great project management, and customer relationships. With advances in machine learning and data science, it's possible to not only predict employee attrition, but to understand the key variables that influence turnover. This will show

### Objectives

- <a href='#setup'>Setting Up</a>
- <a href='#data'>Opening and Visualizing Datasets using `ADS`</a>
   - <a href='#binaryclassifition'>Binary Classification</a>
   - <a href='#data'>The Dataset</a>
   - <a href='#viz'>Visualizing the Dataset</a>
   - <a href='#eda'>Exploratory Data Analysis</a> 
   - <a href='#trans'>Getting and Applying Transformation Recommendations</a> 
- <a href='#model'>Building and Visualizing Models</a>
   - <a href='#automl'>Oracle AutoML</a>
   - <a href='#other_sources'>Models from Other Sources</a> 
- <a href='#eval'>Evaluating Models Using `ADSEvaluator`</a>
- <a href='#explainations'>Explaining How Models Work Using `ADSExplainer`</a>
   - <a href='#adsexplainer'>Using the `ADSExplainer` Class</a>
   - <a href='#global'>Generating Global Explanations</a>
   - <a href='#show'>Showing What the Model Has Learned</a>
        - <a href='#show'>Using `ADSExplainer` for a Custom Model</a>
        - <a href='#pdp'>Feature Dependence Explanations</a>   
   - <a href='#localexplanations'>Generating Local Explanations</a>
- <a href='#ref'>References</a>          
***

### Prerequisites (Optional)

* Experience level: Novice (Python and Machine Learning)
* Professional experience: Some industry experience

This lab assumes you have:
* An Oracle account

In general, the Introduction does not have Steps.

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
