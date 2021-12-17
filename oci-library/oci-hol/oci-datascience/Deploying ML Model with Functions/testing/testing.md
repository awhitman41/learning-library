# Testing the Model Artifact before Saving to the Model Catalog

## Introduction

Always test your model artifact in your notebook session before saving it to the catalog, especially if your function depends on it. 

### Objectives

In this lab, you will:
* Modify the Python Path
* Test the Function’s Handler Defined in func.py

## Task 1: Modify the Python Path

First, modify the Python path and insert the path where the score.py module is located. Then import score and call the predict() function defined in score.py. Load the train dataset and compare the predictions from predict() to the predictions array that you created right after training model. If load_model() and predict() functions are working correctly, you should retrieve the same predictions array. 

```python
# add the path of score.py: 
import sys 
sys.path.insert(0, path_to_rf_artifact)

from score import load_model, predict

# Load the model to memory 
_ = load_model()
# make predictions on the training dataset: 
predictions_test = predict(train, _)

# comparing the predictions from predict() to the predictions array I created above. 
print(f"The two arrays are equal: {np.array_equal(predictions_test['prediction'], predictions)}")
```

The two arrays are identical. 

## Task 2: Test the Function’s Handler Defined in func.py

Next, test the function’s handler defined in func.py. In the following cell, I use the [fnproject Python SDK](https://fnproject.io/tutorials/python/intro/) or fdk to import the context object. The context object is a required argument to the handler() function. It contains attributes such as the application ID, the function ID, the call ID, and the content type of the payload data. 

In this example, we don’t know the application ID, the function ID, or the call ID for my deployed function. These attributes are known after the function has been deployed to Oracle Functions. For now, fill the required parameters with “na”, which is fine for testing purposes. 

I also take the first five rows of my training dataframe (train[:5]), transform them to JSON, and pass them to the handler. This general example shows how a function can score multiple examples at once:

```python
import func
import json 
import logging
from func import handler
from fdk import context
from io import BytesIO

# Example context: 
# app_id, fn_id, call_id, and content_type: 
ctx = context.InvokeContext("na", "na", "na", content_type='application/json')
# Input feature values: 
input_data = {'input':train[:5].to_json()}
json_input = json.dumps(input_data).encode("utf-8")
# calling the Function's handler with my payload: 
resp = handler(ctx, BytesIO(json_input))

# remove most logs printed to screen except critical ones
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
```

You should see the outputs of the two loggers that you defined in func.py: a dictionary containing the values of the input features for the five row examples and a prediction array corresponding to the five predictions (0/1) made by my binary classifier. 

To help with testing the function in production, capture a sample payload of the training data frame. Optionally, save that sample payload file as part of the model artifact file.

```python
# Capturing a sample payload file for testing the Function in production: 

with open(os.path.join(path_to_rf_artifact, "sample-payload.json"),'w') as f:
    json.dump(input_data,f)
```

We’ve tested the files that we need to deploy my model as an Oracle Function. Now, we’re ready to save the model artifact to the model catalog. ADS makes it easy. You can call the save() method of the model artifact that you previously created. If you execute that command, you can see a data frame object with metadata  about your model, confirming that the save operation was successful. 

You can go back to the project page in the Oracle Cloud Infrastructure Console. Under Models, you should see a new entry corresponding to the model that you saved. 

```python
# Saving the model artifact to the model catalog. 

artifact.save(display_name='simple-sklearn-model', 
              description='An initial test to monitor my model in production')
```

**Warning:** If you encounter a __pycache__ error while saving your artifact, simply create a directory called __pycache__ in the artifact directory. You can do it via the terminal with this simple shell command. Execute the save() command again and the error should go away. 

```python
% mkdir __pycache__
```

## Learn More

* [Blog Post- Deploying a Machine Learning Model with Oracle Functions](https://blogs.oracle.com/ai-and-datascience/post/deploying-a-machine-learning-model-with-oracle-functions)
* [Other Related Blog Posts](https://blogs.oracle.com/ai-and-datascience/authors/Blog-Author/COREA7667DA212B34765B4DB91B94737F00E/jean-rene-gauthier)

## Acknowledgements
* **Author** - Jean-Rene Gauthier, Sr Principal Product Data Scientist
* **Contributors** -  Samuel Cacela, Cloud Engineer & Aaron Whitman, Cloud Engineer