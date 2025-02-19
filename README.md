# Insurance Premium Predictor

## Machine Learning Models for Predicting Insurance Premium Estimates

### Description
Hello everyone! This project utilizes Kaggle's Insurance Premium dataset to train a machine learning model. We employ advanced data analysis and feature extraction techniques. Additionally, we implement a **DVC pipeline** for caching stages and log the trained machine learning models and experiments using **MLflow**.

### Instructions for Running Project
- Fork this repository
- Set up a Python or Conda environment to prevent project operations from affecting your local system.
- Install all the project's dependencies specified in **requirements.txt**
  ```
  pip install -r requirements.txt
  ```
  If you use any version of the library other than the one specified in **requirements.txt**, you may encounter errors.

- Open the project's root directory in your terminal and type the command
  
  ```
  pip install -e .
  ```
  
  This adds the project's root directory to the  Python path variable so you can access the root anywhere in the project.
  
- To track the stages present in the project's pipeline using **DVC**, run the following command
  ```
  dvc repro
  ```
   
I hope you all enjoy this project, which was created solely for learning purposes and as a step towards becoming a great Software Developer.
Thank you all for taking the time to look at this project.

### Additional Links
Experiments are tracked on [Dagshub Repository](https://dagshub.com/Shorya777/Insurance_Premium_Prediction).
