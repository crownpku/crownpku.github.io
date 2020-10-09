---
layout: post
comments: true
title: 使用What-If-Tool做机器学习模型评测
published: true
---

**项目代码在：https://github.com/crownpku/Responsible-AI/tree/master/Evaluate_Model**

博主相关系列文章：

* [初探工业界的公平性合规](http://crownpku.com/2020/09/11/%E5%88%9D%E6%8E%A2%E5%B7%A5%E4%B8%9A%E7%95%8C%E7%9A%84%E5%85%AC%E5%B9%B3%E6%80%A7%E5%90%88%E8%A7%84.html)
* [机器学习模型的公平性评测](http://crownpku.com/2020/08/07/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%85%AC%E5%B9%B3%E6%80%A7%E8%AF%84%E6%B5%8B.html)
* [Algorithms Auditing：你的代码公平正义吗？](http://www.crownpku.com/2018/11/14/Algorithms-Auditing-%E4%BD%A0%E7%9A%84%E4%BB%A3%E7%A0%81%E5%85%AC%E5%B9%B3%E6%AD%A3%E4%B9%89%E5%90%97.html)

本文我们来讨论下如何使用谷歌的 What-If-Tool 可视化仪表盘来对机器学习模型进行解释和公平性的评测。

我们使用 Kaggle 上面的保险数据集来构建几个不同版本的机器学习核保模型预测用户的风险因子，然后使用 What-If-Tool 来分享我们一些初步的感想。

## 数据

我们使用生成而非真实的保险数据来构建我们的机器学习核保模型。该数据来自 Brett Lantz 的 Machine Learning with R 书籍。

数据公开在 Github 上：https://github.com/stedy/Machine-Learning-with-R-datasets

数据的内容如下：

![](https://github.com/crownpku/Responsible-AI/raw/master/imgs/insurance_data_pandas.png)

* age: 保单受益人的年龄

* sex: 保单持有人的性别，分为 female 与 male

* bmi: Body mass index，身高体重比，理想值为 18.5 到 24.9

* children: 有几个孩子

* smoker: 是否抽烟

* region: 保单持有人在美国的居住地,包括 northeast, southeast, southwest, northwest.

* charges: 保险为该保单持有人承担的年度平均医疗费用

## 建模

我们把问题分解为几种：

1. 核保模型

    我们简单把保单费用转化为 "Good Risk" 和 "Bad Risk"，将问题转化为一个二分类问题：

     ```python
    def good_risk(row):
        if row['charges'] > 10000:
            return False
        else:
            return True
    
    df_puw['risk'] = df.apply(good_risk, axis=1)
    ```

    然后我们构建三个模型：

        a. Decision Tree 
    
    ```python
    from sklearn.tree import DecisionTreeClassifier
    ```

    b. Logistic Regression
    
    ```python
    from sklearn.linear_model import LogisticRegression
    ```

    c. XGBoost
    
    ```python
    from xgboost.sklearn import XGBClassifier()
    ```

    对于不同模型我们使用相同的预处理步骤，比如对类别特征的 ```SimpleOneHotEncoder()``` 和数值特征的 ```StandardScaler()``` 。

2. 动态定价

    这里我们直接使用保单费用作为预测目标，构建一个 XGBoost 的回归模型：

    ```python
    from xgboost.sklearn import XGBRegressor()
    ```

    我们使用与核保模型相同的预处理方法预处理数据。

## 模型评测

我们现在开启 What-If-Tool，将核保模型中的 Logistic Regression 和 XGBoost 模型与测试数据一起读入。

```python
#fire up the tool
config_builder = (
    WitConfigBuilder(test_examples.values.tolist(), test_examples.columns.tolist())
    .set_custom_predict_fn(customized_prediction_logistic_regression)
    .set_compare_custom_predict_fn(customized_prediction_xgboost)
    .set_target_feature('risk')
    .set_model_type('classification')
    .set_label_vocab(['Bad Risk', 'Good Risk'])
)

WitWidget(config_builder, height=1000)
```

笔者测试的时候，需要使用传统的Jupyter notebook才可以看到仪表盘，而Jupyter lab中是打不开的。

有兴趣的读者还可以将决策树的分类模型以及 XGBoost 的回归模型导入看结果。这里我们专注在有 ```predict_prob()``` 的二分类模型，因为结果最丰富、可玩性最高。

### 数据点可视化

我们可以将标注数据可视化，看下他们在我们模型产生的风险分数里面是如何分布的：

Logistic Regression:

![](https://github.com/crownpku/Responsible-AI/raw/master/imgs/logistic_regression_viz.png)

XGBoost:

![](https://github.com/crownpku/Responsible-AI/raw/master/imgs/xgboost_viz.png)

很明显 XGBoost 在 Good Risks （分数靠近1）和 Bad Risks （分数靠近0）中间相比于 Logistic Regression 有一个更加清晰的分割。

我们可以点击数据点来直接看到保单持有人的详细特征值，以及寻找他们最近的 counterfactual 数据点。最近的 counterfactual （被判别为相反分类的最接近样本）将被选择的数据点与和他在 L1 或者 L2 距离上面最接近但是被分类成另一个类别的数据点做对比。What-If-Tool 同时支持自定义距离函数。

上面的图给出了 counterfactual 的数据点例子。两个保单持有人的特征几乎完全一致，然后一个抽烟一个不抽烟，所以被分类成为不同的风险类别。

我们甚至可以直接在仪表盘中修改特征值，产生新的数据点，然后让模型来做重新预测。这对模型预测行为的探索很有帮助。一些其它的工具还可以支持给出用户提示，如何最小的改变哪些特征的值就可以使自己从 Bad Risk 变成为 Good Risk。

### 公平性

我们的风险分数分类模型的阈值默认是 0.5。这意味着如果模型预测你的风险分数小于0.5你就是 Bad Risk 需要加保费乃至拒保了，而大于 0.5 就是 Good Risk 可以给出保费折扣。

然后这个阈值是可以根据不同的精算需求以及公平性要求来做出调整。

举个例子，我们现在关注在性别这一个特征上，则对于保单持有人可以有以下的公平性考量：

* Single Threshold：所有保单持有人共用一个单独的风险阈值。
* Group Threshold：为男性和女性分别使用各自的风险阈值。
* Demographic parity：被预测为 Good Risks 的人在男性和女性中的比例是一样的。
* Equal opportunity：对于所有标注数据是 Good Risk 的保单持有人，在男性和女性中被正确预测为 Good Risk 的比例是一样的（而不在意 Bad Risk 的预测比例）。
* Equal accuracy：对男性和女性的模型预测准确度是一样的（Good Risk 和 Bad Risk 都要考虑）。

What-If-Tool可以让我们选择如上不同的公平度评测标准，然后优化模型阈值选择：

![](https://github.com/crownpku/Responsible-AI/raw/master/imgs/equal_accuracy.png)

我们可以将工具扩展，可以实时针对不同阈值计算出在测试数据中的保险利润。这也可以帮助精算师进行合理定价。

What-If-Tool还可以有很多其它功能和拓展，比如特征的数值分布，甚至[使用 SHAP 来计算和可视化特征的重要程度](https://github.com/PAIR-code/what-if-tool/blob/master/WIT_COMPAS_with_SHAP.ipynb)等等。有兴趣的读者可以自己探索尝试。



