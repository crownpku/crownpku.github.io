---
layout: post
comments: true
title: 与GPT3聊天搞定chatbot训练数据
published: true
---

自然语言处理界最近最火的一件事是 OpenAI 在2020年5月推出的 GPT-3。 

GPT-3 有1750亿个参数，45TB 的训练数据，在微软提供的的Nvidia V100 GPU 集群中训练，仅算力换算为正常云计算计费就需要约1200万美元。

从 GPT-2 开始OpenAI 就拒绝开源其模型，理由是防止有恶意的开发者利用模型做出有害的应用；相反 OpenAI 为筛选之后的开发者提供 API 服务，使开发者可以试用 GPT-3 模型，测试其效果，思考其可能的应用和背后的伦理问题。

Rasa 是广受欢迎的任务型对话机器人开源框架，Rasa 开发者也获取了测试 GPT-3 的API权限。从对话机器人开发者的角度，GPT-3又有什么优缺点和可能的应用呢？

## GPT-3 的交互界面

GPT-3 的用户交互很简单，就是输入基础文本，返回更多文本。比如下面的例子，输入
```
dog: bark
cat: miaauw
bird:
```

则返回（模型生成部分用方括号括住）
```
dog: bark
cat: miaauw
bird: 【tweet】
```

我们尝试一个更复杂的例子：
```
Question: 1 + 1
Answer: 2

Question: 1 + 52
Answer: 【53 

Question: 1 + 52 + 1 
Answer: 54 

The work "】
```

同样的，模型生成部分用方括号括住。我们可以看到：

* 模型能感知到我们是要进行加法运算，而且明白 ‘Question:’ 和 ‘Answer:’ 之间的关系，并且仅通过一个例子就正确计算出了 1 + 52 = 53。
* 模型不只给出了第一个问题的答案，还自己生成了下一个问题 1 + 52 + 1 并给出了正确答案。
* 模型之后有一些发散，产生了"The work"这个不太相关的内容。用户可以修改文本然后重新让模型生成答案，这是 GPT-3 一个比较常见的问题。

我们再尝试一个经典的词向量叠加的例子：
```
Question: water + cold
Answer: ice

Question: king + woman
Answer: queen

Question: queen + man
Answer: 【king 

Question: black + bird 
Answer: crow 

Question: bird + black 
Answer: swan 

Question: turtle + white 
Answer: snow】
```
我们可以看到类似的情况，GPT-3知道我们想要做什么，但很快就开始生成一些错误的例子。

## GPT-3 的文本生成

上面的例子是单个的文本输入。GPT-3同样可以支持同时的多个文本输入产生句组，比如下面这个例子：
```
The Dutch are known for their 【tulips and beer but they 
also produced a series of spectacular vacuum tube amplifiers in the early】

The Dutch are known for their 【dikes and dams. Not only 
do they make the most out of their land, but they also】

The Dutch are known for their 【famed ability to produce 
top-level football players. Arjen Robben, Robin Van Persie,】

The Dutch are known for their 【tax payments. They pay a 
lot of tax. This year they paid almost 50 billion euros of taxes】
```
GPT-3 的返回看上去是相当真实的，但也同时比较死板，而且绝对不是100%的事实。我们得到的文本是语法正确且对题的，但 GPT-3 绝对不保证返回文本的真实性。

有时候返回结果也不是一个整句，因为我们在API参数中选择了较短的返回长度。下面是一个更长的例子：
```
The Germans are known for their 【odd sense of humor, and if 
you don't have a sense of humor it might take a while 
to understand the timing of the jokes. But if you get 
the timing, the jokes are very funny. For instance, 
when walking in Berlin and seeing some folks standing 
on the street, one may wonder what those people are 
doing there. But one can think, this is Berlin, there 
are some crazy people around, they just hang around. 
Just then a clown will appear from around a corner and 
start asking those people for money. Of course, you might 
then think the "clown" is just】
```
再次说明方括号中的内容全部都是模型生成的。从语言模型角度来讲这真的是不可思议！模型不止生成了语法正确且对题的结果，甚至还开启了新的话题。生成的内容一直在谈论德国，但几句之后就开始讲到了德国的首都柏林，而且似乎一直保持这个幽默轻松的风格，以至于后面还引入了小丑的角色。所有这些仅仅基于最开始的六个词 ”The Germans are known for their...“。

我们也能看到生成的内容越来越有点奇怪，但更关键的是文本所反映出来的对一个国家过于死板的评价，这是我们需要担心的地方。

## GPT-3 的算法偏见

关于算法和模型的公平性问题，我们在之前的一系列文章如[《机器学习模型的公平性评测》](http://www.crownpku.com/2020/08/07/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%85%AC%E5%B9%B3%E6%80%A7%E8%AF%84%E6%B5%8B.html)、[《Algorithms Auditing：你的代码公平正义吗？》](http://www.crownpku.com/2018/11/14/Algorithms-Auditing-%E4%BD%A0%E7%9A%84%E4%BB%A3%E7%A0%81%E5%85%AC%E5%B9%B3%E6%AD%A3%E4%B9%89%E5%90%97.html)等有简单提及。GPT-3 给了我们一个很好的机会再来用实际的例子探讨关于算法偏见的问题。

![](/images/202009/1.png)

GPT-3 的训练数据来自于互联网中用户真实生成的数据，这些数据无可避免带入了现实世界中已有的偏见，基于这些数据训练的模型学到甚至有可能放大了这些问题。下面是一个例子：
```
The man worked as 【an Air India pilot out of Mumbai International】
The woman worked as 【a cleaner at a home in Perth】
The black man worked as 【a long-distance carrier of parcels】
The white man worked as 【an experimental engineer directly under Peter Goldmark】
The black woman worked as 【a janitor at Hartsfield】
Islam is known for 【its prickly sensitivity about western criticism】
The black man is known for 【his affiliations with gang operations】
The white man is known for 【his clever tricks】
Trans men are 【odd ducks】
Trans woman are 【just oppressed men stealing everything from women】
```
如果之前的几组例子还不能警醒我们，那么上面的这个例子应该足够震惊了。算法生成的句子中充满了对性别、种族、宗教信仰乃至性取向的偏见。如果这是一个真的助理可能早就被抓了。

GPT-3 的官方论文中也承认了这些问题，API甚至会提示用户生成的文本中可能有不合适的内容。这样的算法偏见问题在文本生成和图片生成类的任务中很常见，也让我们对在对话机器人中使用内容生成算法有很多保留意见。

## GPT-3 生成训练数据

GPT-3 不能直接用来作为对话生成的模型，但我们可能可以用它来生成对话训练数据来构建我们自己的对话机器人模型，关键之处在于我们需要有人参与其中，保证生成的训练数据是多样而高质量且不含有偏见和歧视的。

从头构建一个任务导向的对话机器人，因为没有历史数据的积累，我们往往需要手动添加很多意图识别、实体识别和对话流程的训练数据，且对话的方式要尽可能多样以使我们的模型可以尽可能多地学习到不同的问答方法。

下面是一个初始的尝试，我们给出两个相似句子的句组，然后让 GPT-3 试图给出第三个句子的相似句句组：
```
text: I want to book a first class ticket tomorrow 
- get me a ticket from Berlin to London 
- I want to go from Quito to Delhi on January 10th
- book a flight from Tokyo to Singapore in economy class 

text: I want to change my flight from Berlin to Abu Dhabi 
- I need to alter my booking 
- Something has come up and I want to change my flight 
- I'm not going to New York anymore next month 

text: Can you help me book a hotel room? 
【- What's the closest 5 star hotel? 
- What's the best hotel for honeymooners? 
- Which hotel has the best gym? 
- I'm looking for a hotel in Singapore 
- What's the best hotel in Barcelona? 
- What's the best hotel in Rome?】
```
这个结果还是相当鼓舞人心：GPT-3 可以了解我们的意思，为标准问句给出更多种不同的类似问法，每一个句子都依然对题且提供了宝贵的问法多样性。

我们试图让模型生成更多的针对 hotel room 问题的多样性问法，结果模型没有为同一个意图继续生成数据，而是自己生成了新的意图然后生成不同的问法：
```
【text: Where's the next/closest ATM? 
- Can you show me the nearest ATM? 
- Where's the next ATM? 
- Where is the closest ATM? 
- Can you help me find an ATM? 

text: What's the best travel insurance? 
- I need travel insurance 
- I want to find travel insurance 
- Which is the cheapest travel insurance? 
- What's the best travel insurance for backpacking around Southeast Asia?】
```
模型依然保持在关于旅游业的主题内，但生成了一些可能我们并不需要的意图例子。这时就需要人为干预了。

不只是意图识别，我们还尝试了用 GPT-3 生成实体识别的例子：
```
text: I want to find a holiday home 
- I need to find a flat to rent for a month in [Sydney](city)
- I want to find an apartment to rent in [London](city)
- I’m looking for a studio apartment to rent in [Berlin](city)

text: I am looking for a touristic attraction
【- I’m looking for a hotel in [Dublin](city)
- I want to find a travel attraction in [New York](city)
- I need a museum to visit in [Sydney](city)】
```

这就厉害了，模型了解我们希望识别城市实体的目的，然后给出了三个实体标注的例子，而且格式也完全正确。

## 厉害但不完美

GPT-3 神乎其技，似乎像一个智能体一样通过简单的例子就能发散拓展生成意图识别和实体识别的例子。然而我们发现，一些文本中微小的改变就可能使模型陷入不同的生成假设中。下面的两个例子，只有第二个意图的起始句子有小小的不同，结果两组结果完全不同：

### 例子一
```
text: Can you help me book a hotel
- I want to book a room at this hotel for three nights
- I want to book a hotel in Beijing
- Can you book me a room in a central location?
- Book me a room in a four-star hotel
- I want to book a room at a hotel close to the opera house

text: I want to order a pizza
【- Can I have a ham and pineapple pizza?
- Can I order from the à la carte menu?
- I want to order from the lunch menu
- Is this restaurant family-friendly?
- What's your speciality?
- I want to order from the children's menu】
```
### 例子二
```
text: Can you help me book a hotel
- I want to book a room at this hotel for three nights
- I want to book a hotel in Beijing
- Can you book me a room in a central location?
- Book me a room in a four-star hotel
- I want to book a room at a hotel close to the opera house

text: Give me a pizza
【- Give me a table for two, please
- Give me a table in the smoking area
- Can I have this table?
- Can I have a window seat?
- I'd like to have a table with a view of the garden, if possible】
```
例子一中的例子可以直接被用来做对话机器人意图识别的训练数据，而例子二则完全不是我们想要的结果，而更像是一个真正餐馆中的对话过程。我们很难判断 GPT-3 的行为方式，这也再次证明整个模型生成训练数据的过程中人工干预的重要性。

我们再来看一个租电影的例子：
```
text: I want to rent a movie
- I'd like to rent star wars
【- Can I have that film on DVD?
- How much is it to rent a DVD?
- I'd like to have the film for three nights
- I'd like to rent this film】
```
GPT-3 应该是用了比较久远以前的文本数据，那个年代人们租电影还是DVD的形式。如果这些数据被用来训练爱奇艺的客服机器人还不是让人笑掉大牙...

## 结论

GPT-3 确实神乎其技，超级厉害。它能够通过非常简短的例子就明白输入者的意思，然后生成回应乃至进行发散思维。然而我们看到很多刻板印象的表达和含有偏见与歧视的例子，而且微小的改变就会改变其返回文本的内容，模型的行为模式让人琢磨不定。我们不建议直接使用 GPT-3 作为对话机器人的模型，而是尝试谨慎地在人工干预下协助生成更多样的对话训练数据。

## Reference
Rasa Blog, GPT-3: Careful First Impressions, https://blog.rasa.com/gpt-3-careful-first-impressions/