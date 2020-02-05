---
layout: post
comments: true
title: Predictive Underwriting：A Technical Case Study
published: true
---

# Introduction

In the last couple of months, my colleagues and me have been working on a project on Predictive Underwriting. The idea of this project is quite simple: you train a machine learning model to give a Risk Score for each insurance customer.

Those with high score are good risks and are eligible for Guaranteed Issued Offer for insurance products. Those with lower scores are eligible for Simplified Issued Offer, with which they need to fill in a simple questionnaire declaring on their past medical history. Those with very bad scores will have to go through the full underwriting process which often includes something like body check.

Traditionally such underwriting process is done with a rule-based system developed by underwriters and actuaries. Predictive Underwriting has the advantages of:

1. Potentially giving more GIO/SIO to customers;

2. Getting more accurate results with historical data;

3. Taking account into more non-traditional features;

4. Being an automated, data-driven and self-evolving underwriting engine.

In this blog, I'm about to cover the data science/technical part of this project. I will not cover the very standard parts in data science like the common techniques for feature engineering or handling missing data. Instead I will focus more on the **experience**, the lessons learned from successes and more importantly failures.


# Infrastructure

It's not trivial to talk about infrastructure first as this is the place where all the work is done.

## Environment

For my own preference, the ideal environment should be a mainstream Linux environment (for example Ubuntu) with proper computational resource and internet connection for installing the required packages. It can be either a physical machine or an environment in the cloud. It should also be able to connect to the internal database, so data can be easily manipulated and extracted directly.

In reality, we got a physical Windows workstation, fortunately with internet connection. This caused some problems that we must work around.

Installing Python environment and Jupyter notebooks on Windows is not trivial, not to mention all the IT firewalls that usually block the automatic way of installing packages. Instead we must download the compiled packages from PyPI and copy them to the specific folder in the workstation and configure the path. If you are using R language, with software like Rstudio this should be easier in Windows environment.

We also could not access the internal database directly from the workstation. Colleagues from client side were kind enough to extract data based on our needs. However, this gives us less flexibility and control on the data source.

## No Data Out

Client has very strict policy that no data can be transferred out of their environment. Ideally this issue can be solved with a secure remote working setup, or with advanced secure machine learning techniques like Federated Learning. 

In our case we can only work onsite in their office with the specific workstation and a Windows account that the specific colleague from client side can log into. This is very expensive as in the past months we had to fly to client office to do the actual heavy coding. I still remember one day the colleague on client side who has the account and password left the office, and I needed to use the bathroom and had to ask another colleague to keep moving the mouse so that we don't get locked out of the system…

Later after several rounds of working together, we gained some trust from client and we started using the remote-control function in Skype meetings to do some remote coding. This worked pretty well for parameter tuning and debugging, but the connection is not perfect, and it is too laggy to do any heavy coding from scratch. So, every time we visited client I was still under some pressure trying to finish all the heavy coding. 

# Data, Data, Data!

There are three most important things for a machine learning project: data, data and data. We need data and we need good data. This comes to the following data qualities:

1. Large dataset. For our case this is not a problem as client has accumulated quite a lot of data in their IT system.

2. Wide range of feature set. For our case we have features from demographics data, underwriting data and claim data. Ideally more features like social network, fitness data etc. can be added in and this will be a unique advantage of machine learning models compared to traditional UW dataset.

3. Enough label data. This is the tricky one as we need risk labels on existing policyholder to teach the algorithm to calculate the risk score for new customers. Usually we do not have a out-of-box label for this and we will cover it in a later session how we managed to get our labels. 

## Data Sources

Historical data, usually because of legacy issues, are stored in different format at different places. It's an important first step to understand the data we can get. In our case, demographic features, underwriting features and claim features from health rider data are from different databases. Below is a picture of the relationship between them.

We need to then understand what part of the data will be used as training data with possible labels, and what part of the data can be used as the potential customer base for the model to predict on. Our plan is to build three models:

* Model 1: With full feature set of Demographic + UW + Health (Claim)

* Model 2: With feature set of only Demographic + Health (Claim)

* Model 3: With feature set of only Demographic

From Model1 to Model 3, as we are missing more and more features, we expect the model performance will decrease, however the model will also cover larger customer base when it comes to prediction of the risk scores.

It's also important to distinguish between the training data and prediction data here. While customer base data for prediction is different, we do use the same set of training data with labels for all the 3 models. The difference between the 3 models is only on the feature set during the training process.

![](/images/202002/1.png)

## Label Data

To get the best label quality, our label data is generated from the group of customers with full feature set, namely Demographic, Underwriting and Claim features.

Traditionally the labels, namely the risk label on each customer, are generated from some rules. In our case the rules are the following:

* Previous underwriting decision on the customer is substandard.

* Previous claimed disease from this customer is ranked as high severity level.

* Loss ratio (claim amount over premium) on the customer is more than a certain level.

Besides the rules, we also want to introduce as much as possible of the experience of underwriters into the labelling process. Ideally the underwriters should label each of the customer again by looking at all the features, and with millions of policyholders this becomes unpractical. We designed the following unsupervised learning/manual labelling process to tackle this problem: 

1.	Cluster the policyholders into a comfortable number of clusters. We used elbow method to decide on the number of clusters. Below is a plot of the result from the elbow method, and we use 70 as our number of clusters. We also got feedbacks that certain features like the claim disease severity are very important features, so we put higher weights on them when we calculate the distance function during the clustering process (K-mean in our case). 

![](/images/202002/2.png)
 
2.	For each of the cluster we plot the distribution of the common features underwriters use for risk evaluation. This can be done either by plotting some interactive visualizations directly in Jupyter notebooks, or by using other tools like PowerBI or Tableau.

3.	Manual labelling of good/bad risks by the underwriters on the clusters. There are also cases where it's not clear for certain clusters to be labelled as good or bad risk, and those clusters will be labelled as "exclude" so we will not use them as part of our training data.

After this process, each customer in the cluster will have a cluster label as "Good" or "Bad". This result will be combined with label generated from the rules to get the final "Ground-truth" label of Good or Bad risk as illustrated in the picture below:  

![](/images/202002/3.png)

Note that with this mechanism of generating the ground-truth label, there will be some customers who get inconsistent labels from the rules and the clustering result, and we will exclude them from our training dataset.

# Modelling

With label data, we can go into the modelling part to training a supervised machine learning model to predict the risk scores for customers.

## Data Splitting

Customer data is usually time dependent data. Although it is not necessary and many times not appropriate to use time series analysis to work on this data, we still need to be aware of the time dependent nature of it, and process data accordingly.

The simplest way for handling this time dependent data is to use the same time period for generating both features and labels. This is how the traditional rule based underwriting process goes, and we are predicting the risk of the customers "at this moment". However even we have introduced the clustering process for labelling the risks, having the same time period for features and labels will inevitably result in an algorithm trying to imitate the rules themselves.

The second way is to divide the time periods for features and labels. Features imitate the "past" evidence and labels indicate the "future" risk. Below is an example: during the training process, features are generated from Jan.2014 to Oct. 2017, and labels (with rules and clustering/labelling) are generated from Nov.2017 to Aug.2019. We train the model this way so that it is using the past data to predict risk score for the future. Thus, when we do the model prediction, we will use all the history data as the feature, and predict the risk scores for the next 2 years from Aug.2019. This is the method we used for our case. Note that as policy will span in different years, some of the numerical features like claim amount will need to be normalized.

![](/images/202002/4.png)

There are also other ways of splitting the data on the time line. For example, this split can be customer dependent. We can use the first several years after one policy was bought to generate the labels and use data before this policy purchase as features.

Now we get the correct features and labels, we need to divide the data into training set, validation set and testing set for the machine learning process. Training set is for training the model itself, validation set is used to monitor the training process to get the best hyperparameters (mostly to avoid over-fitting), and testing set is to finally evaluate the model performance.

We use the rule of thumb 20/80 split for this. 20% of the data is used as testing set, 16% (20% of the left data) is used as validation set, and the rest 64% is used as training set. One small trick is once we decide on the model hyper-parameters and ready to go production, we will use the same hyper-parameters to retrain the model on all the data set (100% training set) a gain a few percentages of model improvement.

![](/images/202002/5.png)

Again, there are also other ways of splitting data for training and testing purposes. One of them is to again split on the time line, so that we train a model with the historical data as the training set and test it in the recent one year as the testing set to check its predictive power for the future.

## Handling Imbalanced Dataset

Our case is a binary classification problem on good and bad risks. We find that the label data is very imbalanced with way more good risks than bad risks (ratio around 30:1).

One possible way of tackling this is to use technical like over-sampling or down-sampling to balance the dataset. We don't want to use down-sampling as it will decrease significantly our data volume for good risks. Over-sampling methods like SMOTE will create some new synthetic data points imitating the bad risks, which we finally decided also not to use as the synthetic data points are not real cases anyways.

What we did is changing the weights in the costing function so that it will give a bigger punishment if it mislabels a bad risk. We use the Gradient Boosted Tree as our classification method, with the famous python package XGBoost. We simply set the parameter "scale_pos_weight" as the ratio between number of good risks and number of bad risks to achieve this. AUC is used for evaluation of the model during the training process.

One thing to note is that by changing the weight in the costing function, the risk score we finally get will not be the accurate "probability" of a good risk. However, the ranking of the risk scores is still valid, and as later you will see we will tune the threshold for GIO and SIO, so it will not be an issue.

If you are interested, please check out the XGBoost documentation here on handling imbalanced dataset:

https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html#handle-imbalanced-dataset

## Evaluating Performance

For a binary classification problem, typically AUC is the best way to evaluate the model performance. In our case the AUC is around 0.8:

![](/images/202002/6.png)
 
We need to carefully discuss the model performance with business stakeholders and interpret it in business sense. AUC is not very straight forward in this sense.

First thing we did is to plot the most important features from our model. We found that age, BMI, policy length, claim history and sum insured are among the most important features, which makes business sense.

We then plotted the confusion matrix for our model. This is before threshold tuning so the threshold is by default 0.5 for good and bad risks. This brings very interesting resonance from business. Which metrics are important from business perspective? For example, suppose we get a confusion matrix below:

![](/images/202002/7.png)
 
We think that two basic metrics are important:

1. Recall for Bad Risk = TN/(TN+FP). A bad value of this means we mislabel many bad risks as good risks and will bring financial risk to business.

2. Precision for Good Risk = TP/(TP+FP). A bad value of this means there will be a lot of missed opportunities.

We will be trying to find a balance between these two metrics when we do the threshold tuning. 

## Threshold Tuning

Once we have built the model and make the risk score predictions on the potential customer base, we will need to decide on the threshold for the SIO and GIO offers. 

Besides getting the balance between potential financial risk and missed business opportunities, the threshold tuning is mainly a back testing on the data to find the get the sensible threshold for SIO and GIO from an actuarial point of view. This analysis is mainly done on the testing data so that we imitate what will happen for the real predictions. Important attributes to consider are loss ratio, age distribution, claim experience etc. Below is an example of such analysis in excel:

![](/images/202002/8.png)
 
In fact, this analysis gives the actuaries a lot of confidence in deploying our predictive modelling, as clearly with the increase of the score the metrics indicate better health and lower risks.

# Bottom Line

Predictive Underwriting is a big topic covering much more topics than this blog can include, things like secure data exchange and modelling, model interpretation, deployment and integration, model updates etc. This blog is simply some personal experience from a client project as a technical case study.
Feel free to drop an email at guan_wang@swissre.com if you have any questions regarding this blog.
