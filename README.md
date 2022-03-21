# AzureML and Data Science Overview Workshop

The objective of this workshop is to work through a basic end-to-end flow for a data scientist starting to work with AzureML to work interactively with a model and data, train it on a cluster, deploy to an endpoint and test it. Use Visual Studio Code to work interactively with the training script on a Compute Instance. Use the Studio GUI and v2 CLI to setup your environment and compute, submit jobs to a cluster, and create scoring endpoints. 

The AzureML docs and samples will help you figure out syntax. The scenario has been tested and you’ll see that the product is pretty good overall. Things are not consistent and some experiences are still rough. With the exception of some internal private preview features, this is what our customers see.

This is hands-on and not a demo or tour of the product. The knowledge required of python, Linux, AzureML and Azure is minimal. There are some hints below to point you in the right direction. Search the Web, read docs and samples, and reach out to team members for help if you get stuck.


## 0. Getting Started

You need access to an Azure subscription. If it’s a shared subscription, the owner should provide you with a resource group and assign you as owner so you can wear an IT hat and create a workspace and required resources. 

This repo has the python scripts and data files you need to get started. 

Hint: You need compute quota for AzureML in the region for the workspace for the VM family you want to use. 4 cores of any D series should be fine to create a compute instance, cluster, and realtime endpoint. 


## 1. Train and test Locally
First step is to train a model locally to make sure it works. Create a Compute Instance, and work from Visual Studio Code. The script uses MLFlow logging and will log to an AzureML workspace without having to submit a run. Take a look at the experiments in Studio.

Once you’ve trained a model locally, try the score.py script to test it locally on a subset of data. 

Hints: For local training, the train.py script is setup to use the training data csv file in the same directory. It writes the model file to a new folder, deleting an existing folder with the same name if found. You will need to configure your Python environment with the packages needed for the training script. Use the conda.yml file, then select and activate the environment using the Python: Select Interpreter command from the Command Palette.


## 2. Experiment and Explore the Data
Now we get into the science part to experiment on how we could improve the model. Edit the train.py file and change [param] to True.

Train the model again, run the predict script, and compare the outputs. Look at the two runs in the Studio and compare metrics between the two models model. What do you see?

The data imbalance hyperparameter in LightGBM helps in the case where the training data has more on-time flights and so biased the outcome in the model. This is something a data scientist needs to be aware of.

Try different ways to explore the data. Open it in Excel, important it as a tabular dataset, or use Python tools. Think about how we can help with this.

What metrics do you think are important to measure quality of this prediction? Try optimizing for a different metric.


## 3. Train in Cloud
Now that you’ve confirmed the training code works, train the model in the cloud using an AzureML job. Use the train.py and flightdelayweather_ds_clean.csv from the repo (but imagine that you’re training on petabytes of data). Try this from both the Studio and the v2 CLI. You can create a train.yaml file from Visual Studio Code.

Look at differences between running locally and logging metrics and submitting a run. One example is the training script is saved as as artifact when submitting a run. This is helpful for reproducibilty. 

Hints: Your training run needs an environment with the package dependencies and the CSV file in a dataset as input to train with. There is more than one way to handle this. Remember that the local training run you did assumed a local data file instead of passing this in as a parameter as --data. 


## 4. Create Managed Real-Time Endpoint
After training a model, create a real-time managed endpoint and test us using the sample JSON file in the repo. Try this with the Studio and the v2 CLI. 

TODO: Need sample JSON to test with

You could also write a script or app to use the endpoint. 

## 5. Create Managed Batch Endpoint
Next create a batch endpoint. There is a scoring CSV in the repo. If you do this in the Studio, you will see that a pipeline was created. Look at the results and how many flights are predicted to be on-time vs delayed. 

Hint: Use the predict_data.csv for the batch scoring job.


## 6. Reflect and Discuss
This was a simple exercise, but used the breadth of AzureML for ML Pros working with python scripts using Studio, CLI, VS Code or other tools. You had to create a basic workspace (no VNET today), create and use computes, train models, look at metrics, create and test endpoints, and experiment with the training code and data.

Was it easy to get started and figure out what to do? Did you need to look at documentation or samples? We’re things consistent across the Studio, v2 CLI and SDK? Did you have to troubleshoot any error messages? Was the Studio experience intuitive? What are you going to personally improve from what you experienced in this workshop?


## 7. Bonus: Improve the Model
Now think about this as a Kaggle competition. Have some fun and try out different ways to improve the model quality. You could look at AutoML, Flaml, or other approaches. Share your best model with the team. 


## 8. Bonus: Create a component
Create a component from the train.py file and use it in a pipeline to re-train a model that is then used for batch scoring. Try this using the CLI and Studio.


## If you're really stuck
Look in the <TODO> branch for more help.
