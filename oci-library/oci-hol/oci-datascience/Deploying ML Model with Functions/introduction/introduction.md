# Introduction

## About this Workshop

Notebook originally downloaded from [this blog](https://blogs.oracle.com/ai-and-datascience/post/deploying-a-machine-learning-model-with-oracle-functions).

You can download a [notebook (.ipynb) version of this workshop post](https://objectstorage.us-ashburn-1.oraclecloud.com/p/9mRu1h5oqBO7P8iSnrW0hIhoJUBm-mkpzoYL3VZPXG8JBMS0iOLIu26x26AlGiuZ/n/bigdatadatasciencelarge/b/hosted-datascience-docs/o/blogs/e2n_fn/blog-post-e2e-fn.ipynb). I recommend that you use and run the notebook in your notebook session, instead of copying code snippets in a blank notebook.

**To use this notebook, use Kernel:**
`conda env:generalmachinelearningforcpusv2_0`

How to set up kernel:
- `File` > `New Launcher`
- Click `Environment Explorer`
- Search for the Conda environment listing using the Conda environment's slug name: `generalmachinelearningforcpusv2_0` (or search for `General Machine Learning for CPUs`, and find the listing for `v2.0`)
- On the Conda environment listing, follow the `Install` instructions`General Machine Learning for CPUs`

**In this workshop, we will show you how to build and train a model in a notebook session of the [Oracle Cloud Infrastructure (OCI) Data Science service](https://docs.cloud.oracle.com/en-us/iaas/data-science/using/data-science.htm) and deploy that model with [Oracle Functions](https://docs.cloud.oracle.com/en-us/iaas/Content/Functions/Concepts/functionsoverview.htm%22%20%5Cl%20%22Overview_of_Functions).**

Before we get started, let's make sure we upgrade the client `oci` to the latest version. Some of the capabilities around logging and log search are only available in the recent versions of `oci`. In a cell of your notebook, execute: 

```python
%%bash 

pip install oci --upgrade
pip install fdk
pip install pandas
pip install numpy
pip install scikit-learn
pip install cloudpickle
```

### Objectives

In this workshop, you will learn how to:
* Train a Model in a Notebook Session
* Save the Model to the Model Catalog 
* Testing the Model Artifact Before Saving to the Model Catalog 
* Deploying the model as an Oracle Function through Oracle Cloud Infrastructure Cloud Shell
* Invoking the Deployed Function

### Prerequisites

Note that these steps assume that you are a tenancy admin whose user has privileges to manage all resources in the tenancy.

- Make sure that this notebook is included as part of a Dynamic Group (e.g. DSDynamicGroup) with a matching rule:\
    `ALL {resource.type = 'datasciencenotebooksession', resource.id = '<ID of this Notebook Session>'}`

- Make sure that there are policy statements that enable dynamic group to access the necessary resources. E.g.:\
    `Allow service datascience to use virtual-network-family in compartment <name of compartment with networking resources>`
    
     `Allow service datascience to manage data-science-family in compartment <name of compartment with Data Science resources>`

    `Allow service datascience to use functions-family in compartment <name of compartment with Functions resources>`
   
    `Allow dynamic-group DSDynamicGroup to use virtual-network-family in compartment <name of compartment with networking resources>`

    `Allow dynamic-group DSDynamicGroup to manage data-science-family in compartment <name of compartment with Data Science resources>`
    
    `Allow dynamic-group DSDynamicGroup to use functions-family in compartment <name of compartment with Functions resources>`

## Learn More

* [Blog Post- Deploying a Machine Learning Model with Oracle Functions](https://blogs.oracle.com/ai-and-datascience/post/deploying-a-machine-learning-model-with-oracle-functions)
* [Other Related Blog Posts](https://blogs.oracle.com/ai-and-datascience/authors/Blog-Author/COREA7667DA212B34765B4DB91B94737F00E/jean-rene-gauthier)

## Acknowledgements
* **Author** - Jean-Rene Gauthier, Sr Principal Product Data Scientist
* **Contributors** -  Samuel Cacela, Cloud Engineer & Aaron Whitman, Cloud Engineer

