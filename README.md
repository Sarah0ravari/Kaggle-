# Kaggle-
Santander Customer Satisfaction

In this project, we describe our approach for building data models to predict which customers are happy customers. The raw dataset and the problem statement are taken from the Santander Customer Satisfaction competition on Kaggle.

Please download the test and train data from https://www.kaggle.com/c/santander-customer-satisfaction/data

The dataset - train.csv was obtained from Kaggle.com.
  I did this project as part of my MATH 445.[ Statistical Foundations to Machine Learning].
  The detailed final report on the analysis is available in this repository.
  The scprit.R has the R codes used to analyse the data using the methods that we learned in math 445 class will cover concepts of linear models and prediction 
  models including generalized linear models, supervised and unsupervised learning such as classification techniques and clustering. It also includes missing data
  techniques and concepts of time series analysis.


### Importance of the problem

Assessing customer satisfaction is the most important aspect of staying competitive in banking industry.
Dissatisfied customer are more likely to leave and usually it happens without any prior warning. In order to
address this problem, Santander Bank provided an anonymized dataset for identification of customer satisfaction at kaggle.com.
The problem faced by the bank is that dissatisfied customers usually leave without prior notice. This makes it
a difficult job for the bank to anticipate customer dissatisfaction. Hundreds of parameters that may influence
customer satisfaction are given in the dataset and a prediction model is to be implemented in identifying
dissatisfied customers.

### Statement of objective

This prediction model will help in identification of the customers whose probability of dissatisfaction increases
to the point of concern, so that a proactive service approach can be adopted to avoid loss in business. The
dataset obtained is of a higher dimension, containing thousands of data points and hundreds of predictors. It
is highly imbalanced in favor of satisfied customers, unsatisfied customers constitute a mere 3.95% of the
total data points.

### Methods used

As a first step, cleaning of data is performed after an exploratory analysis. Exploratory analysis evaluates the
dataset for missing and values, constant, duplicated and correlated predictors. Over sampling techniques
were employed to balance the training dataset, oversampling doesn’t suffer from any information loss and is
preferred over under sampling technique. Sampling techniques are used to remove major class bias in the
modelling process. Random forest and Fselector packages were employed for feature selection to identify
significant variables. Models were trained on balanced dataset obtained by sampling techniques and tested
on imbalanced original test data. Models were then evaluated on the basis of AUC (area under curve) in ROC
analysis, and the corresponding results were reported. Modelling techniques of logistic regression, LDA,
random forest, artificial neural network, gradient boosted machine and xgboost are applied.

### Key results

- using Cross Validated, on the imbalanced training data.
- The model gave the second best performance with 0.8328.

### Implication of results

The modern and sophisticated ensemble models work very well with the imbalanced
dataset to give both better prediction performance and fast computational speeds.
