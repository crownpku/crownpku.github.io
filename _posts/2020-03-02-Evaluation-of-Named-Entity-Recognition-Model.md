---
layout: post
comments: true
title: Evaluation of Named Entity Recognition Model
published: true
---

# Introduction

Named Entity Recognition, or NER, is a task where a model will try to recognize the named entities from the raw corpus. NER is usually considered as a sequence labelling task. With the input data as a sentence or a paragraph, the model will try to label:

1. Start position and end position of the entity

2. Entity type

There are many ways to train such a NER model, for example a rule-based system, a Conditional Random Field (CRF) model or more recently deep learning models. This article tries to focus on the evaluation of such NER models. This is crucial as the performance metrics is important not only for the final stage model evaluation but also for the model training process acting as the optimization function.

# Types of Errors

Let's give an example here to illustrate the different types of errors model can make.

Suppose we are trying to extract the city name from the following sentence:

"I love Singapore and it has a lot of good restaurants."

The ground truth will be:

" I love Singapore (city) and it has a lot of good restaurants."

The model gives the following possible cases:

1. Model gives the exact same results as the ground truth:

"I love Singapore(city) and it has a lot of good restaurants."

2. Model invents an entity that doesn't exist in the ground truth:

"I love Singapore(city) and it has a lot of good restaurants(city)."

3. Model misses an entity in the ground truth:

"I love Singapore and it has a lot of good restaurants."

4. Model gives the wrong entity type:

"I love Singapore(person) and it has a lot of good restaurants."

5. Models gets the wrong entity boundary:

"I love Singapore(city) and it has a lot of good restaurants."

6. Model gives the wrong entity type and gets the wrong entity boundary:

"I love Singapore(person) and it has a lot of good restaurants."

The complexity of the above cases requires a more sophisticated evaluation method than simply calculating precision/recall/F1 score as we do for simple classification tasks.

# Evaluation Metrics for NER

We are following the SemEval'13 (International Workshop on Semantic Evaluation) notion of evaluation metrics for NER. 

In their original notion, they specifically distinguish if it allows the NER model to make the entity type correct or not. We think that entity type is very important. Imagine we run something like the SpaCy model and only get the ORG types as the target NER results. If entity type is wrongly labeled, we will not be able to get the target company or organization entities. With this in mind we simplify the evaluation method into two categories, namely Strict Mode and Loose Mode. For the original definition of SemEval'13 please check out the nice blog in [1].

* Strict Mode: Exact entity boundary and entity type are matched.

* Loose Mode: Entity type is matched; there are overlapping part on the entity boundary for ground truth and model results.

Now if we go back to the error cases:

* Case 1 is a successful match for both Strict and Loose Mode. 

* Case 2, for both Strict and Loose Mode, Singapore(City) is a successful match, while restaurant(city) is a failed match.

* Case 3 is a failed match for both Strict and Loose Mode.

* Case 4 is a failed match for both Strict and Loose Mode.

* Case 5 is a failed match for Strict Mode, but a successful match for Loose Mode.

* Case 6 is a failed match for both Strict and Loose Mode.

Now we cover all the error cases, for both Strict and Loose Mode we can calculate the following:

* True Positive (TP): Model and Ground Truth Match

* False Positive (FP): The entities Model gets that are wrong or not in Ground Truth

* False Negative (FN): The entities Model doesn't get but that are in the Ground Truth.

So, we can now calculate a reasonable precision/recall/F1 score for both Strick and Loose Mode:

* Precision = TP/(TP+FP) which describes the percentage of correct entities found by the NER model.

* Recall = TP/(TP+FN) which describes the percentage of ground truth entities that are retrieved from the NER model.

* F1 Score = 2 · Precision · Recall/(Precision + Recall) which is the harmonic mean of precision and recall.

This way we get the model evaluation metrics as the F1 score for both Strict and Loose Mode. 

# Conclusion

As we can see, a poor precision means the NER model may find a lot of the entities but many of them are not really in the ground truth, while a poor recall means many of the entities in the ground truth are not discovered by the NER model. Many times we need to find a balance between precision and recall and F1 score is a good way to do that.

More importantly, we look at all the possible types of errors we can get in the NER task, and design the Strict and Loose mode to define a successful or a failed match from the NER model considering the entity types and word start/end boundaries that fits our needs (always value the entity types, but may choose Loose model if we are OK with partially wrong entity boundaries).

# Reference

[1] http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

[2] https://nlpers.blogspot.com/2006/08/doing-named-entity-recognition-dont.html
