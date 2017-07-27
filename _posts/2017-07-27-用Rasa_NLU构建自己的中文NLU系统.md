---
layout: post
comments: true
title: 用Rasa NLU构建自己的中文NLU系统
published: true
---

自然语言理解（NLU）系统是问答系统、聊天机器人等更高级应用的基石。基本的NLU工具，包括实体识别和意图识别两个任务。

已有的NLU工具，大多是以服务的方式，通过调用远程http的restful API来对目标语句进行解析完成上述两个任务。这样的工具有Google的[API.ai](http://api.ai), Microsoft的[Luis.ai](http://luis.at), Facebook的[Wit.ai](http://wit.ai)等。刚刚被百度收购的[Kitt.ai](http://kitt.ai)除了百度拿出来秀的语音唤醒之外，其实也有一大部分工作是在NLU上面，他们很有启发性的的Chatflow就包含了一个自己的NLU引擎。

(话说，现在.ai的域名都好贵呢...其实是安圭拉的国家域名，这个小岛就这么火了...)

以上这些工具，一边为用户提供NLU的服务，一边也通过用户自己上传的大量标注和非标注数据不断训练和改进自己的模型。而对于数据敏感的用户来说，开源的NLU工具如[Rasa.ai](http://rasa.ai)就为我们打开了另一条路。更加重要的是，这样的开源工具可以本地部署，自己针对实际需求训练和调整模型，据说对一些特定领域的需求效果要比那些通用的在线NLU服务还要好很多。

我们在这里就简单介绍使用Rasa NLU构建一个本地部署的特定领域的中文NLU系统。


Rasa NLU本身是只支持英文和德文的。中文因为其特殊性需要加入特定的tokenizer作为整个流水线的一部分。我加入了jieba作为我们中文的tokenizer，这个适用于中文的rasa NLU的版本代码在[github](https://github.com/crownpku/rasa_nlu_chi)上。


## 施工中...


### 语料获取及预处理


### MITIE模型训练


### 构建rasa_nlu语料和模型


### 搭建本地rasa_nlu服务


### Rasa UI界面 



