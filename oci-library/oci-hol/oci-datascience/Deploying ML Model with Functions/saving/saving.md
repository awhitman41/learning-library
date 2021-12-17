# Saving the model in the model catalog

## Introduction

To save the model in the catalog, we use the [ADS library](https://docs.cloud.oracle.com/en-us/iaas/tools/ads-sdk/latest/index.html) and its [prepare_generic_model()](https://docs.cloud.oracle.com/en-us/iaas/tools/ads-sdk/latest/user_guide/modelcatalog/modelcatalog.html%22%20%5Cl%20%22generic-model-approach) function, which probably is the easiest method. First, create a temporary local directory where we’re going to store the model artifact files: 

### Objectives

In this lab, you will:
* Saving the model in the model catalog
* Adding loggers to handler() 


## Task 1: Saving the model in the model catalog

First, create a temporary local directory where we’re going to store the model artifact files: 

```python
from ads.common.model_artifact import ModelArtifact
from ads.common.model_export_util import prepare_generic_model
import os

#Replace with your own path: 
path_to_rf_artifact = f"ML_Model_Artifact"
if not os.path.exists(path_to_rf_artifact):
    os.mkdir(path_to_rf_artifact)
```

I use ADS `prepare_generic_model()` to create all the necessary templatized files that are part of the model artifact. You still need to modify each one of the files in the artifact to fit your particular use case. In the next few cells I will go each one of the files I modified and create. Note that all the files in the target artifact directory will be compressed and shipped to the model catalog as your model artifact. 

```python
artifact = prepare_generic_model(path_to_rf_artifact, force_overwrite=True, data_science_env=True)
```

In the latest release of the notebook session environment, ADS also generates all the Oracle Functions artifact files (func.py, func.yaml, requirements.txt) by default. Using these files, I later deploy my model as an Oracle function. 

First, serialize the random forest classifier and save it to disk. I use joblib to save my model to disk, which is the preferred way for [scikit-learn models](https://scikit-learn.org/stable/modules/model_persistence.html). 

```python
from joblib import dump

dump(clf, os.path.join(path_to_rf_artifact, "rf.joblib"))
```

Now that we have a serialized model object in our artifact directory, modify the func.py file, which contains the definition of the Oracle Functions handler (handler()) function. Oracle Function calls the handler function.  



## Task 2: Adding loggers to handler()

In the following cell, I write a new version of func.py. Executing this cell overwrites the template that ADS provides as part of the model artifact. 

There are a few differences to the template. I import the Python logging library and define the model-prediction and model-input-features loggers. I use these loggers to capture the model predictions and the model input features for each call made to the function. With these loggers, I monitor how my predictions and features distributions change over time. Those log entries are captured and stored in the [Logging service](https://docs.cloud.oracle.com/en-us/iaas/Content/Logging/Concepts/loggingoverview.htm#loggingoverview). 

Then, I add some more data transformations in handler(). You can achieve a similar outcome by adding those transformations to the body of predict() in score.py. 

```python
%%writefile {path_to_rf_artifact}/func.py

import io
import json

from fdk import response
import sys
sys.path.append('/function')
import score
import pandas as pd
model = score.load_model()

# Importing and configuring logging: 
import logging
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# configuring logging: 
# For model predictions: 
logger_pred = logging.getLogger('model-prediction')
logger_pred.setLevel(logging.INFO)
# For the input feature vector: 
logger_input = logging.getLogger('model-input-features')
logger_input.setLevel(logging.INFO)

def handler(ctx, data: io.BytesIO=None):
    try:
        input = json.loads(data.getvalue())['input']
        logger_input.info(input)
        input2 = json.loads(input)
        input_df = pd.DataFrame.from_dict(input2)
        prediction = score.predict(input_df, model)
        logger_pred.info(prediction)
    except (Exception, ValueError) as ex:
        logger_pred.info("prediction fail {}".format(str(ex)))

    return response.Response(
        ctx, response_data=json.dumps("predictions: {}".format(prediction)),
        headers={"Content-Type": "application/json"}
    )
  ```

Next, modify the requirements.txt file. ADS generates a template for requirements.txt that provides a best guess at the dependencies necessary to build the Oracle Function and run the model. In this case, modify the template and add the following dependencies on scikit-learn version 0.23.2: 

```python
%%writefile {path_to_rf_artifact}/requirements.txt

cloudpickle
pandas
numpy
fdk
scikit-learn
```

Last, modify the inference script score.py, which loads the model to memory, and call the predict() method of the model object. 

By default, ADS generates this file assuming that you’re using cloudpickle to read the model serialized object. I modified score.py to use joblib and left the definition of predict() intact. 

```python
%%writefile {path_to_rf_artifact}/score.py

import json
import os
from joblib import load

"""
   Inference script. This script is used for prediction by scoring server when schema is known.
"""


def load_model():
    """
    Loads model from the serialized format

    Returns
    -------
    model:  a model instance on which predict API can be invoked
    """
    model_dir = os.path.dirname(os.path.realpath(__file__))
    contents = os.listdir(model_dir)
    model_file_name = "rf.joblib"
    # TODO: Load the model from the model_dir using the appropriate loader
    # Below is a sample code to load a model file using `cloudpickle` which was serialized using `cloudpickle`
    # from cloudpickle import cloudpickle
    if model_file_name in contents:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), model_file_name), "rb") as file:
            model = load(file) # Use the loader corresponding to your model file.
    else:
        raise Exception('{0} is not found in model directory {1}'.format(model_file_name, model_dir))
    
    return model


def predict(data, model=load_model()) -> dict:
    """
    Returns prediction given the model and data to predict

    Parameters
    ----------
    model: Model instance returned by load_model API
    data: Data format as expected by the predict API of the core estimator. For eg. in case of sckit models it could be numpy array/List of list/Panda DataFrame

    Returns
    -------
    predictions: Output from scoring server
        Format: { 'prediction': output from `model.predict` method }

    """
    assert model is not None, "Model is not loaded"
    # X = pd.read_json(io.StringIO(data)) if isinstance(data, str) else pd.DataFrame.from_dict(data)
    return { 'prediction': model.predict(data).tolist() }
```

## Learn More

* [Blog Post- Deploying a Machine Learning Model with Oracle Functions](https://blogs.oracle.com/ai-and-datascience/post/deploying-a-machine-learning-model-with-oracle-functions)
* [Other Related Blog Posts](https://blogs.oracle.com/ai-and-datascience/authors/Blog-Author/COREA7667DA212B34765B4DB91B94737F00E/jean-rene-gauthier)

## Acknowledgements
* **Author** - Jean-Rene Gauthier, Sr Principal Product Data Scientist
* **Contributors** -  Samuel Cacela, Cloud Engineer & Aaron Whitman, Cloud Engineer
