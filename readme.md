# 介绍

## 描述

基于keras的文本情感分析

## 前言

这个项目是随便写写的,可能不是很好,不做过多维护

# 使用方法

- 训练:
  ```python
  from HKSA import HKSA
  train_data={'this is a data':'this is a tag','this is also a data':'this is also a tag'}
  model = HKSA()
  model.create_from_data(train_data)
  model.train(train_data, batch_size=1, epochs=4)
  ```
- 预测:
  ```python
  import os.path

  from HKSA import HKSA
  
  model_name = 'weibo_senti_100k.csv.txt'
  model = HKSA()
  model.load(os.path.join('model', model_name + '.json'))

  predict_list=['this is a data','this is also a data']
  result = model.predict(predict_list)
  for i in range(len(predict_list)):
      print(result[i], predict_list[i])
  ```
- 评估:
  ```python
  from HKSA import HKSA
  data={'this is a data':'this is a tag','this is also a data':'this is also a tag'}
  model = HKSA()
  model.create_from_data(data)
  model.train(data, batch_size=1, epochs=4)
  ```
