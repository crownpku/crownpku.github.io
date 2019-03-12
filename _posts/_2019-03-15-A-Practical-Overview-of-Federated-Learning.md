---
layout: post
comments: true
title: A Practical Overview of Federated Learning
published: true
---

## Introduction

Federated Learning originated from [an academic paper](https://pmpml.github.io/PMPML16/papers/PMPML16_paper_20.pdf) in NIPS 2016 [1] and a [follow-up blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) [2] in 2017, both published by Google. The idea is that Google wants to train its own input method on its Android phones called "Gboard" but does not want to upload the sensitive keyboard data from their users to Google’s own servers. Rather than uploading user’s data and training models in the cloud, Google lets users train a separate model on their own smartphones (thanks to the neural engines from several chip manufacturers) and upload those black-box model parameters from each of their users to the cloud and merge the models, update the official centralized model, and push the model back to Google users. This not only avoids the transmission and storage of user’s sensitive personal data, but also utilize the computational power on the smartphones (a.k.a the concept of Edge Computing) and reduce the computation pressure from their centralized servers.

![](/images/201901/2.png)

When the concept of Federated Learning was published, Google mentioned data privacy and utilized the the simple but effective parameter average for model ensembling, but the main focus was on the transmission of models as the upload bandwidth of mobile phones is usually very limited. One possible reason is that similar engineering ideas have been discussed intensively in **Distributed Machine Learning**. Slave-master structure is widely used and Parameter Server is so powerful that model parameters or even gradients during training process can be computed and shared synchronously or asynchronously. The focus of Federated Learning was thus on the “engineering work” with no rigorous distributed computing environment, limited upload bandwidth and slave nodes as massive number of users.

Data privacy is becoming a more and more important issue, and lots of relating regulations and laws have been taken into action by authorities and governments. The companies that have been accumulating tons of data and have just started to make value of it now have their hands tightened. On the other hand, market competition is increasingly intense and all companies value and secure their own data from sharing with the others. Information islands kill the possibility of cooperation and mutual benefit. Everyone is looking for a way to break such prisoner dilemma while complying with all the regulations. **Federated Learning is soon being recognized as a great solution for encouraging collaboration while respecting data privacy**.

## Categorization

Professor YANG Qiang from The Hong Kong University of Science and Technology [extends Federated Learning to three types](https://www.chainnews.com/articles/769415855789.htm): Horizontal Federated Learning, Vertical Federated Learning and Federated Transfer Learning [3][4]. **Horizontal Federated Learning** applies to circumstances where we have a lot of overlap on features but only a few on instances. This refers to the Google Gboard use case and models can be ensembled directly from the edge models. **Vertical Federated Learning** refers to where we have many overlapped instances but few overlapped features. An example is between banks and online-retailers. They both have lots of same users but each own their own features and labels data. Vertical Federated Learning merges the features together to create more powerful feature space for machine learning tasks, and uses homomorphic encryption to provide protection on data privacy. **Federated Transfer Learning** uses Transfer Learning to improve model performance when we have neither much overlap on features nor on instances. Model ensembling, homomorphic encryption and transfer learning are all well studied directions, but such categorization extends the concept of Federated Learning and clarify the specific solutions under different use cases.

## Platforms and Tools

Todo

### FATE

Todo

### Tensorflow FL

Todo

## Use cases

Todo

### In Input Methods (Google)

Todo

### In Healthcare

https://www.technologyreview.com/s/613098/a-little-known-ai-method-can-train-on-your-health-data-without-threatening-your-privacy/

### In Insurance

Todo

## Outlook

Todo

### Hierarchy FL (Users -> Primary Insurers -> Reinsurers)

Todo

### Regulation/Audition solution

Todo

### Heterogeneous Feature Space for Federated (Transfer) Learning

Todo

### Incentives/Credit Allocation

Todo
