---
layout: post
comments: true
title: Annotation Tools and Beyond
published: true
---


## Introduction

I will mainly talk about the technical details of an open source project [Chinese Annotator](https://github.com/crownpku/Chinese-Annotator) I've been working on and some thoughts around it. Before that, I'd like to emphasize the context of why we need such annotation tools.


## Why Annotation Tools

Most machine learning problems are supervised learning. Be it recommentation systems, image recognition or natrual language processing, we all probably need label data for building good models and making correct predictions.

In some lucky cases, such label datasets are automatically generated. For example, in recommendation systems, people natrually generate label data by clicking on an ad on the webpage, adding stuff in their shopping card on Amazon, or listening to songs on Spotify.

In some other cases, we can find smart tricks to generate label data. For example, when we want to build an entity recognition or relation extraction model from the text, we can actually follow the so called "Distant Supervision" idea by crawling entity and relation data from the internet and using them to train the model. Though such data may be noisy, we find some deep learning models quite robust.

However, there can be cases when neither automatically generated label data nor smart tricks are available. There are already quite some image recognition annotation tools out there, for example bla bla and bla.

![](image of image annotation tools)

There are also tons of companies offering human labour for generating label data. For example there are lots of commercial "human captcha recognition" services in China for crawling need, and even Amazon has their [Mechanical Turk](https://www.mturk.com/) human labelling service which they call "Human intelligence through an API".

You can imagine that there are many cases where not all the data can be easily shared with those commercial services. First problem is data privacy. As the most valuable asset companies cannot simply give data away. Second problem is the lack of expert knowledge. Some labelling task requires deep expert knowledge to make a judge on the labels, and such is not always suitable for oursourcing the task.

This is where we need annotation tools that are easy to deploy, intuitive to use, and effective to generate label data.

## Annotator for Chinese Text Corpus

Most tasks in Natrual Language Processing are supervised learning problem. There are sequence labelling problems such as Toeknization and Named Entity Recognition, classification problems such as relation extraction, sentiment analysis and intention recognition. All those problems needs label data for training the model. With deep learning conquering almost all the problems, DL based NLP models are expecially data thirsty.

Things are better for English language. Many large scale corpus are out there for people to research on such as SQuAD Reading Comprehension corpus from Stanford. For Chinese, such open source corpus are way less, making it quite diffcult to transfer the state-of-art technologies developed with English to Chinese. On the other hand, for some vertical context such as health, finance, legal and public security, there are bunch of special entities and needs. We cannot use general models trained on google news or wikipedia dump in those special contexts.

Traditional annotation methods are complex and cumpsy. We have to label multiple times for "Swiss Re", "Swiss Reinsurance" and "Swiss Re Group". Such process needs large amount of repetitive human labours.

Can we build a Chinese text annotator, so that

1. Labelling process has intelligent algorithms behind it so that we minimize human labours.

2. Labelling UI is friendly, intuitive and easy to use.

The answer is yes!

![](image of english version of software structure)

### Intelligent Labelling with Active learning

### Data Pipeline

### Modular Design

### More than an Annotator

**Data Manager**

**Model Manager**

**Prediction Service**

Obviously, at end of the day, this project will be a **Full Pipeline Machine Learning Tool**.

## One More Thing

As you may know I joined Swiss Re Hong Kong not long ago as a Data Scientist. Swiss Re is the second largest reinsurance company globally, and it's amazing how emoumrous data and projects we have at hand to fully embrace the world of automation and intelligence. 

However what amazed my most during my still-short time at Swiss Re is the working style, which Swiss Re calls "Own the way you work". We are encouraged to work from home when appropariate. And more interestingly, in our office at Wanchai each and every employee from interns to directors does not own a permanent desk. Instead we are given a fancy suitcase called "Hot Box", and we are asked to carry such suit case and sit anywhere we want in the spacious two floors of office. Especially for data scientists like me, we are encouraged to switch desks frequently so that we can meet more colleagues, understand their needs, and see if we can help with data science.

![](Picture of office view)

Of course at Swiss Re we have Microsoft products and network proxies everywhere, and for data scientists this is sometimes painful. But the "own the way you work" style is already too cool to be true at such a big company. This inpires me to share my working style on the Chinese Annotator Project.


## Reference

This article intends to be an English re-writing of my previous two blogs [构想：中文文本标注工具](http://www.crownpku.com/2017/11/09/%E6%9E%84%E6%83%B3-%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E6%A0%87%E6%B3%A8%E5%B7%A5%E5%85%B7.html) and [分布式异步协作让世界更美好](http://www.crownpku.com/2017/11/28/%E5%88%86%E5%B8%83%E5%BC%8F%E5%BC%82%E6%AD%A5%E5%8D%8F%E4%BD%9C%E8%AE%A9%E4%B8%96%E7%95%8C%E6%9B%B4%E7%BE%8E%E5%A5%BD.html). I'm re-writing in English for sharing with my international colleagues at Swiss Re as well as to the English speaking community.