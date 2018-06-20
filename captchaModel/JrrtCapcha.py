# encoding:utf-8
import random
import numpy as np
import string

from PIL import Image, ImageFont, ImageDraw, ImageFilter

from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adagrad
from keras.models import Model, load_model
import warnings
warnings.filterwarnings('ignore')
import os

from keras.callbacks import ModelCheckpoint, LearningRateScheduler


'''
    A default vocab implementation and base class, to provide random letters and numbers.
'''
class Vocab():
    def __init__(self):
        # self.vocab = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.vocab = string.digits + string.ascii_lowercase + string.ascii_uppercase
        self.size = len(self.vocab)
        indices = range(self.size)
        self.index = dict(zip(self.vocab, indices))
    # return random string by given length
    def rand_string(self, length):
        # if len(vocab) == 0 raise exception
        return "".join(random.sample(self.vocab, length))
    # get symbol (char in vocabulary) by its ordinal
    def get_sym(self, idx):
        # if idx >= len(self.vocab) raise exception
        return self.vocab[idx]
    # given a symbol, return its ordinal in given vocabulary.
    def get_index(self, sym):
        return self.index[sym]
    # given 'abc', return [10, 11, 12]
    def to_indices(self, text):
        return [self.index[c] for c in text]
    # given [10, 11, 12], return 'abc'
    def to_text(self, indices):
        return "".join([self.vocab[i] for i in indices])
    # given '01', return vector [1 0 0 0 0 0 0 0 0 0 ... 0 \n 0 1 0 0 0 0 ... 0]
    def text_to_one_hot(self, text):
        num_labels = np.array(self.to_indices(text))
        n = len(text)
        categorical = np.zeros((n, self.size))
        categorical[np.arange(n), num_labels] = 1
        return categorical.ravel()
    # translate one hot vector to text.
    def one_hot_to_text(self, onehots):
        text_len = onehots.shape[0] // self.size
        onehots = np.reshape(onehots, (text_len, self.size))
        indices = np.argmax(onehots, axis = 1)
        return self.to_text(indices)


"""
 Captcha 用来模拟生成 "今日头条"的验证码
"""
class Captcha(object):
    '''
    size: width, height in pixel
    font: font family(string), size (unit pound) and font color (in "#rrggbb" format)
    bgcolor: in "#rrggbb" format
    '''

    def __init__(self, size, font, bgcolor, length=4):
        # todo: add param check and transform here
        self.width, self.height = size
        self.font_family, self.font_size, self.font_color = font
        self.bgcolor = bgcolor
        self.len = length
        self.vocab = Vocab()
        self.font = ImageFont.truetype(self.font_family, self.font_size)

    def get_text(self):
        return self.vocab.rand_string(self.len)

    # by default, draw center align text
    def draw_text(self, str):
        dr = ImageDraw.Draw(self.im)
        font_width, font_height = self.font.getsize(str)
        # don't know why, but for center align, I should divide it by 2, other than 3
        dr.text(((self.width - font_width) / 3, (self.height - font_height) / 3), str, fill=self.font_color,
                font=self.font)

    def draw_background(self):
        pass

    def transform(self):
        params = [1 - float(random.randint(1, 2)) / 100,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 10)) / 100,
                  float(random.randint(1, 2)) / 500,
                  0.001,
                  float(random.randint(1, 2)) / 500
                  ]
        self.im = self.im.transform((self.width, self.height), Image.PERSPECTIVE, params)

    def filter(self):
        self.im.filter(ImageFilter.EDGE_ENHANCE_MORE)

    # by default, add no noises
    def add_noise(self):
        pass

    def get_captcha(self):
        self.im = Image.new("RGB", (self.width, self.height), (self.bgcolor))
        self.draw_background()
        str = self.get_text()
        self.draw_text(str)
        self.add_noise()
        self.transform()
        self.filter()
        return self.im, str


"""
 JrttCapcha 继承自 Captcha, 重写了 add_noise 方法，给验证码图片加了一些"噪点"
"""
class JrttCaptcha(Captcha):
    def __init__(self, size=(120, 30), font=("DejaVuSerif.ttf", 20, "#0000ff"), bgcolor=(255, 255, 255),
                 dot_rate=0.05):
        Captcha.__init__(self, size, font, bgcolor)
        self.dot_rate = dot_rate

    def add_noise(self):
        # add lines
        nb_lines = random.randint(1, 2)
        dr = ImageDraw.Draw(self.im)
        for i in range(nb_lines):
            # 避免begin和end太靠近，导致生成的干扰线太短
            begin = (random.randint(0, self.width) / 2, random.randint(0, self.height) / 2)
            end = (random.randint(self.width / 2, self.width), random.randint(0, self.height))
            dr.line([begin, end], fill=(0, 0, 0))
        # add dots
        for w in range(self.width):
            for h in range(self.height):
                if random.randint(0, 100) / 5 <= self.dot_rate:
                    dr.point((w, h), fill=(0, 0, 0))

    def draw_text(self, str):
        display_text = [" "] * (len(str) * 2 - 1)
        for i in range(len(str)):
            display_text[i * 2] = str[i]
        super(JrttCaptcha, self).draw_text(str)


# 使用 ImageGenerator 来实时增强数据集
def JrttCaptchaGenerator(batch_size, path=None):
    # to determine dimensions
    # cap = captcha.JrttCaptcha()
    cap = JrttCaptcha()
    img, text = cap.get_captcha()
    shape = np.asarray(img).shape
    vocab = Vocab()
    while (1):
        X = np.empty((batch_size, shape[0], shape[1], shape[2]))
        Y = np.empty((batch_size, len(text) * vocab.size))
        for j in range(batch_size):
            img, text = cap.get_captcha()
            #img.save(path + text + ".jpg")
            X[j] = np.array(img) / 255
            Y[j] = vocab.text_to_one_hot(text)
        yield X, Y


batch_size = 32
ocr_shape = (30, 120, 3) # height, width, channels
nb_classes = 62

# 构建神经网络
def create_cnn_model():
    inputs = Input(shape=ocr_shape, name="inputs")
    conv1 = Conv2D(32, (3,3), name='conv1')(inputs)
    relu1 = Activation('relu', name='relu1')(conv1)
    conv2 = Conv2D(32, (3,3), name='conv2')(relu1)
    relu2 = Activation('relu', name='relu2')(conv2)
    pool1 = MaxPooling2D(pool_size=(2,2), padding='same', name='pool1')(relu2)
    conv3 = Conv2D(64, (3,3), name='conv3')(pool1)
    relu3 = Activation('relu', name='relu3')(conv3)
    pool2 = AveragePooling2D(pool_size=(2,2), name='pool2')(relu3)

    fl = Flatten()(pool2)
    fc1 = Dense(nb_classes, name='fc1')(fl)
    drop =Dropout(0.25, name="dropout1")(fc1)
    fc21 = Dense(nb_classes, name='fc21', activation='softmax')(drop)
    fc22 = Dense(nb_classes, name='fc22', activation='softmax')(drop)
    fc23 = Dense(nb_classes, name='fc23', activation='softmax')(drop)
    fc24 = Dense(nb_classes, name='fc24', activation='softmax')(drop)

    merged = merge([fc21, fc22, fc23, fc24], mode='concat', name="merged")

    model = Model(inputs=inputs, outputs=merged)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    return model

def lr_scheduler(epoch, mode='power_decay'):
    lr_base = 0.001
    lr_power = 0.9
    epochs=50

    lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    print 'learnig rate: ', lr
    return lr




def train_model(epochs=50):
    model_file = 'Jrtt_capcha_model.h5'
    if os.path.exists(model_file):
        print('Load captcha model from %s' % model_file)
        model = load_model(model_file)
    else:
        model = create_cnn_model()

    model.summary()

    save_model_callback = ModelCheckpoint('Jrtt_capcha_model.h5', monitor='acc', save_best_only=True)
    learning_rate_callback = LearningRateScheduler(lr_scheduler)

    model.fit_generator(JrttCaptchaGenerator(batch_size), steps_per_epoch=100, epochs=epochs,
                        callbacks=[save_model_callback])



import sys
if __name__ == "__main__":
    args = sys.argv[1:]
    nb_epochs = 50 # default is 50 epochs
    if len(args) > 1:
        nb_epochs = int(args[0])


    print('Will process %d epochs.' % nb_epochs)
    train_model(nb_epochs)
