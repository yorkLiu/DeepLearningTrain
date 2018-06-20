# 基于Keras的验证码识别

## 【今日关条验证码识别】模型 
- 代码 [JrrtCapcha.py](JrrtCapcha.py)
- 训练好的模型[Jrtt_capcha_model.h5](Jrtt_capcha_model.h5)
  > 可以使用以下方法来加载已有的模型
  ```python
    from keras.models import load_model
    model_file = 'Jrtt_capcha_model.h5'
    model = load_model(model_file)
  ```