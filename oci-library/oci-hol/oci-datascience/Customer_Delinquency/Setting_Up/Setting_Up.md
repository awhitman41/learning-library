#Setting Up

## Introduction

This labs walks you through how to connect our Autonomous Data Warehouse (ADW) to Oracle's Data Science platform.

Estimated Time: -- 10 minutes 

### Prerequisites

This lab assumes you have:
* An Oracle account
* An ADW (assuming you have one created, if not visit https://oracle.github.io/learning-library/data-management-library/autonomous-database/shared/adb-quickstart-workshop/freetier/?lab=adb-provision-conditional)

### Objectives

In this lab, you will:
* Set up our environment 
* Connect to the ADW where our data is staged

## Set Up

### Open Data Science Platform

```python
import warnings
warnings.filterwarnings('ignore')

import os
import cx_Oracle

import ads
import oci
import logging
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

from ads.dataset.factory import DatasetFactory
from ads.automl.provider import OracleAutoMLProvider
from ads.automl.driver import AutoML
from ads.automl.provider import OracleAutoMLProvider
from ads.catalog.model import ModelCatalog
from ads.catalog.project import ProjectCatalog
from ads.common.model_artifact import ModelArtifact
from ads.common.data import ADSData
from ads.common.model import ADSModel
from ads.dataset.factory import DatasetFactory
from ads.evaluations.evaluator import ADSEvaluator
from ads.explanations.explainer import ADSExplainer
from ads.explanations.mlx_global_explainer import MLXGlobalExplainer
from ads.explanations.mlx_local_explainer import MLXLocalExplainer
from category_encoders.ordinal import OrdinalEncoder
```

```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

## Connect to the ADW where our data is staged

```python
%env TNS_ADMIN=/home/datascience/"ADW_USER"_wallet
%env ADW_SID=_"ADW_USER_high"
%env ADW_USER="ADW_USER"
%env ADW_PASSWORD="ADW_PASSWORD"

!echo exit | sqlplus ${ADW_USER}/${ADW_PASSWORD}@${ADW_SID}

uri=f'oracle+cx_oracle://{os.environ["ADW_USER"]}:{os.environ["ADW_PASSWORD"]}@{os.environ["ADW_SID"]}'
```

env: TNS_ADMIN=/home/datascience/jmadw_wallet
env: ADW_SID=jmadw_high
env: ADW_USER=admin
env: ADW_PASSWORD=Welcome#1234

SQL*Plus: Release 19.0.0.0.0 - Production on Tue Mar 2 18:13:35 2021
Version 19.6.0.0.0

Copyright (c) 1982, 2019, Oracle.  All rights reserved.

Last Successful login time: Tue Mar 02 2021 02:45:02 +00:00

Connected to:
Oracle Database 19c Enterprise Edition Release 19.0.0.0.0 - Production
Version 19.5.0.0.0

SQL> Disconnected from Oracle Database 19c Enterprise Edition Release 19.0.0.0.0 - Production
Version 19.5.0.0.0

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
