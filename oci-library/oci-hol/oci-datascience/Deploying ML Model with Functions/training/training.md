# Training a Model in a Notebook Session

## Introduction

To get started I train a simple binary classifier using `scikit-learn` in a notebook session of the Oracle Cloud Infrastructure Data Science service. The business problem it itself and the quality of the model do not really matter in this case and the model is only used for illustrative purposes. In fact any binary classifier would do just fine. 

### Objectives

In this lab, you will:
* Use Function to Create a Balanced Synthetic Binary Classification Dataset
* Train the Simple Binary Classifier

## Task 1: Use Function to Create a Balanced Synthetic Binary Classification Dataset

Use the `sklearn` `make_classification()` function to create a balanced synthetic binary classification dataset and you are going to use it to train a random forest classifier. The model takes in eight numerical features labelled `feat1`,...,`feat8`. 

You can do the same by launching a notebook session in Oracle Cloud Infrastructure Data Science service and execute the following cell in your notebook. You can use [resource principals](https://blogs.oracle.com/datascience/resource-principals-data-science-service) to authenticate to the model catalog, object storage, and Oracle Functions. It is recommended to go over the `getting-started.ipynb` notebook yourself to get your session set up with resource principals. 

```python
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np 
import pandas as pd 
import ads 

# Using resource principal to authenticate when using the model catalog: 
ads.set_auth(auth='resource_principal') 

# Here we create a simple (balanced) binary classfication dataset with 8 features, 
# and a sample of 10K rows. 
X, y = make_classification(n_samples=10000, n_features=8, n_informative=4,
                           random_state=0, shuffle=True)

# convert to a pandas dataframe for easy manipulations: 
train = pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4', 
                                 'feat5', 'feat6', 'feat7', 'feat8'])
target = pd.Series(y)
```

```python
target.value_counts()
```

From the `value_counts()` call you can see that the dataset is balanced which is what I want. 


## Task 2: Train the Simple Binary Classifier

Next step is to train a simple binary classifier. In this case, I use a `RandomForestClassifier` also available in `scikit-learn`: 

```python
# training the random forest classifier from scikit-learn: 
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(train, target)
```

You can check the accuracy of the model on the training dataset using the following command: 

```python
predictions = clf.predict(X)
diff = np.abs(predictions - y)
print(f"accuracy (train):{1.0 - sum(diff) / len(predictions)}")
```

I get an accuracy of about 0.75 which is fine. You should get similar results in your notebook. 

Now, we have a working binary classifier. Save this model to the [model catalog](https://docs.cloud.oracle.com/en-us/iaas/data-science/using/manage-models.htm). 

## Learn More

* [Blog Post- Deploying a Machine Learning Model with Oracle Functions](https://blogs.oracle.com/ai-and-datascience/post/deploying-a-machine-learning-model-with-oracle-functions)
* [Other Related Blog Posts](https://blogs.oracle.com/ai-and-datascience/authors/Blog-Author/COREA7667DA212B34765B4DB91B94737F00E/jean-rene-gauthier)

## Acknowledgements
* **Author** - Jean-Rene Gauthier, Sr Principal Product Data Scientist
* **Contributors** -  Samuel Cacela, Cloud Engineer & Aaron Whitman, Cloud Engineer