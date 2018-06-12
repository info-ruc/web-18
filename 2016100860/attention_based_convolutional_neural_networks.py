
# coding: utf-8

# In[1]:




# In[2]:

#每个用户历年的指标（n_times，n_feature）组成一个矩阵，k个邻居组成k个矩阵
#类似于3个通道的图片(RGB)
#加上Conv1D层以及globalmaxpooling得到（1，n_filter）可以考虑用不同的stride步长 或设置长度的maxpooling 
#主用户历年体检指标同样构成一个矩阵经处理得到（1，n_filter）
#最后把两个分支一起作为mlp的输入 得到血压值
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[3]:

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)    


# In[4]:

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import json

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 


# In[5]:

def readFromJson():
        with open('./TJ_180403.json', 'rb') as lf:
            rec = json.load(lf, encoding='bytes')
        return rec


# In[6]:

TJ_dict = readFromJson()


# In[7]:

from produce_single_var_data import *


# In[8]:

data=produce_single_var_data(3, TJ_dict,trainf = "train_3_bph_180517_2.8.txt", testf ="test_3_bph_180517_2.8.txt")


# In[63]:

TJ_dict['987322']


# In[9]:

x_train, y_train, x_test, y_test, train_context, test_context = data


import tensorflow as tf


# In[18]:

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, GlobalAveragePooling3D,Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D,Conv1D,GlobalAveragePooling1D
from keras.datasets import imdb
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Lambda, Layer, Masking
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard


# In[19]:

n_features = 25
maxlen = 10
n_context = 4
# Convolution
kernel_size = 2
filters = 32
#pool_size = 4

# Training
batch_size = 128
epochs = 20


# In[74]:

#只有一种卷积长度 且 仅仅对年份卷积
def cnnOfYear(x_train, x_test, y_train, y_test, c_train, c_test, epoch):
    m_model = Sequential()
    m_model.add(Conv1D(64, input_shape=(maxlen, n_features),  kernel_size=(kernel_size,), padding='valid', strides=(1,), activation='relu'))
    m_model.add(Conv1D(32, kernel_size=(2,), padding='valid', strides=(2,), activation='relu'))
    m_model.add(Conv1D(16, kernel_size=(2,), padding='valid', strides=(2,), activation='relu'))
    #m_model.add(Conv1D(8, kernel_size=(2,), padding='valid', strides=(2,), activation='relu'))
    
    #m_model.add(Conv2D(filters, input_shape=(maxlen, n_features, 1), data_format='channels_last', kernel_size=(kernel_size, n_features), padding='valid', strides=(1, n_features), activation='relu'))
    #m_model.add(Conv2D(8,data_format='channels_last', kernel_size=(kernel_size, 1), padding='valid', strides=(1, 1), activation='relu'))
    
    m_model.add(GlobalAveragePooling1D())
    m_input = Input(shape=(maxlen, n_features))
    c_input = Input(shape=(n_context,))
    en_m = m_model(m_input)
    print(en_m.shape)
    x = keras.layers.concatenate([en_m, c_input])
    print(x.shape)
    mlp1 = Dense(10, activation='relu')(x)
    mlp1 = Dense(5, activation='relu')(mlp1)
    output = Dense(1)(mlp1)
    model = Model(inputs = [m_input, c_input], outputs=output)
    print(output.shape)
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    model.fit([x_train, c_train], y_train, batch_size=256, epochs=epoch, shuffle=True, 
                validation_data=([x_test, c_test], y_test),
                callbacks=[TensorBoard(log_dir='./tmp/log_cnn')])
    score = model.evaluate([x_test, c_test], y_test)
    print(model.summary())
    print(score)
    return score


# In[75]:

cnnOfYear(x_train, x_test, y_train, y_test, train_context, test_context, 130)


# In[28]:

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, GlobalAveragePooling3D
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.datasets import imdb
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Lambda, Layer, Masking
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from sklearn import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Embedding, Lambda, Activation
from keras.layers import Dense, Input, Flatten, RepeatVector, Permute, Reshape
from keras.layers import merge, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


# In[67]:

def cnn_1att(x_train, x_test, y_train, y_test, c_train, c_test, epoch):
    np.random.seed(1)
    m_input = Input(shape=(maxlen, n_features))
    m_dense = TimeDistributed(Dense(64, activation='relu'))(m_input)
    print(m_dense.shape)
    attention = TimeDistributed(Dense(1, activation='relu'))(m_dense) # try diff act
    print(attention.shape)
    attention = Flatten()(attention)
    print(attention.shape)
    attention = Activation('softmax')(attention) # try different activations
    print(attention.shape)
    attention = RepeatVector(25)(attention)
    print(attention.shape)
    attention = Permute([2, 1])(attention)
    print(attention.shape)
    m_att = merge([m_input, attention], mode='mul')
    print(m_att.shape)
    #sentence_attention = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(26,))(sent_representation)
    #m_att = Reshape((10, 25, 1))(m_att)
    print(m_att.shape)
    m_model = Sequential()
    m_model.add(Conv1D(64, input_shape=(maxlen, n_features),  kernel_size=(kernel_size,), padding='valid', strides=(1,), activation='relu'))
    m_model.add(Conv1D(32, kernel_size=(2,), padding='valid', strides=(2,), activation='relu'))
    m_model.add(Conv1D(16, kernel_size=(2,), padding='valid', strides=(2,), activation='relu'))
    m_model.add(GlobalAveragePooling1D())
    
    n_context=4
    c_input = Input(shape=(n_context,))
    
    en_m = m_model(m_att)
    print(en_m.shape)
    meg = keras.layers.concatenate([en_m, c_input])
    mlp1 = Dense(10, activation='relu')(meg)
    mlp1 = Dense(5, activation='relu')(mlp1)
    
    output = Dense(1)(mlp1)
    
    
    
    model = Model(inputs = [m_input, c_input], outputs=output)
    print(output.shape)
    
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mse','mae'])
    model.fit([x_train, c_train], y_train, batch_size=256, epochs=epoch, shuffle=True,validation_data=([x_test, c_test], y_test),
                callbacks=[TensorBoard(log_dir='./tmp/log_cnn_year')])
    score = model.evaluate([x_test, c_test], y_test)
    print(model.summary())
    print(score)
    return score


# In[76]:

cnn_1att(x_train, x_test, y_train, y_test, train_context, test_context, 120)


# In[99]:

def cnn_2att(x_train, x_test, y_train, y_test, c_train, c_test, epoch):
    m_input = Input(shape=(maxlen, n_features))
    
    att_ = TimeDistributed(Dense(n_features, input_shape = (n_features,), activation='softmax', name='attention_probs'))(m_input)
    m_dense = merge([m_input, att_], output_shape=(maxlen, n_features), name='attention_mul', mode='mul')
     
   
    print(m_dense.shape)
    
    attention = TimeDistributed(Dense(1))(m_dense) # try diff act
    print(attention.shape)
    attention = Flatten()(attention)
    print(attention.shape)
    attention = Activation('softmax')(attention) # try different activations
    print(attention.shape)
    attention = RepeatVector(25)(attention)
    print(attention.shape)
    attention = Permute([2, 1])(attention)
    print(attention.shape)
    m_att = merge([m_dense, attention], mode='mul')
    print(m_att.shape)
    #sentence_attention = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(26,))(sent_representation)
    #m_att = Reshape((10, 26, 1))(m_att)
    print(m_att.shape)
    
    m_model = Sequential()
    m_model.add(Conv1D(64, input_shape=(maxlen, n_features), kernel_size=(kernel_size,), padding='valid', strides=(1,), activation='relu'))
    #m_model.add(Conv1D(64, input_shape=(maxlen, n_features),  kernel_size=(kernel_size,), padding='valid', strides=(1,), activation='relu'))
    m_model.add(Conv1D(32, kernel_size=(2,), padding='valid', strides=(2,), activation='relu'))
    m_model.add(Conv1D(16, kernel_size=(2,), padding='valid', strides=(2,), activation='relu'))
    #m_model.add(Conv1D(8, kernel_size=(2,), padding='valid', strides=(2,), activation='relu'))
    
    m_model.add(GlobalAveragePooling1D())
    
    n_context=4
    c_input = Input(shape=(n_context,))
    
    en_m = m_model(m_att)
    print(en_m.shape)
    meg = keras.layers.concatenate([en_m, c_input])
    mlp1 = Dense(10, activation='relu')(meg)
    mlp1 = Dense(5, activation='relu')(mlp1)
    
    output = Dense(1)(mlp1)
    
    
    
    model = Model(inputs = [m_input, c_input], outputs=output)
    print(output.shape)
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    model.fit([x_train, c_train], y_train, batch_size=256, epochs=epoch, shuffle=True, validation_data=([x_test, c_test], y_test),
                callbacks=[TensorBoard(log_dir='./tmp/log_cnn_hir')])
    score = model.evaluate([x_test, c_test], y_test)
    print(model.summary())
    print(score)
    return score


# In[100]:

cnn_2att(x_train, x_test, y_train, y_test, train_context, test_context, 100)


# In[42]:

#attr = ['15#一般检查#收缩压', '15#一般检查#舒张压', '1#血常规#白细胞计数（WBC）', '1#血常规#血红蛋白含量（Hb）', '1#血常规#红细胞计数（RBC）', '1#血常规#红细胞比积（HCT）', '1#血常规#平均红细胞体积（MCV）', '1#血常规#平均红细胞血红蛋白含量（MCH）', '1#血常规#平均红细胞血红蛋白浓度（MCHC）', '1#血常规#血小板计数（PLT）', '1#血常规#平均血小板体积（MPV）', '1#血常规#血小板分布宽度（PDW）', '1#尿常规#尿葡萄糖（GLU）', '1#尿常规#尿胆原（URO）', '1#尿常规#尿酸碱度（PH）', '1#尿常规#尿比重（SG）', '1#尿常规#尿蛋白质（PRO）', '1#血脂检测#甘油三酯（TG）', '1#血脂检测#总胆固醇（TC）', '1#血脂检测#高密度脂蛋白胆固醇（HDL-C）', '1#血脂检测#低密度脂蛋白胆固醇（LDL-C）', '1#肾功能检测#肌酐（Cr）', '1#肾功能检测#尿酸（UA）', '1#糖尿病检测#空腹血糖（FPG）', '体重指数', '年龄']


# In[101]:

def cnn_1att_indice(x_train, x_test, y_train, y_test, c_train, c_test, epoch):
    np.random.seed(1)
    m_input = Input(shape=(maxlen, n_features))
    t_x = Permute([2, 1])(m_input)
    print(t_x.shape)
    t_x = Reshape((25, 10))(t_x)
    print('transpose shape:', t_x.shape)
    m_dense = TimeDistributed(Dense(64, activation='relu'))(t_x)
    print(m_dense.shape)
    attention = TimeDistributed(Dense(1, activation='relu'))(m_dense) # try diff act
    print(attention.shape)
    attention = Flatten()(attention)
    print(attention.shape)
    attention = Activation('softmax')(attention) # try different activations
    print(attention.shape)
    attention = RepeatVector(10)(attention)
    print(attention.shape)
    attention = Reshape((10, 25))(attention)
    print('original_shape:', attention.shape)
    m_att = merge([m_input, attention], mode='mul')
    print(m_att.shape)
    #sentence_attention = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(26,))(sent_representation)
    #m_att = Reshape((10, 25, 1))(m_att)
    print(m_att.shape)
    m_model = Sequential()
    m_model.add(Conv1D(filters, input_shape=(maxlen, n_features), kernel_size=(kernel_size,), padding='valid', strides=(1,), activation='relu'))
    m_model.add(GlobalAveragePooling1D())
    
    n_context=4
    c_input = Input(shape=(n_context,))
    
    en_m = m_model(m_att)
    print(en_m.shape)
    meg = keras.layers.concatenate([en_m, c_input])
    
    mlp1 = Dense(10, activation='relu')(meg)
    mlp1 = Dense(5, activation='relu')(mlp1)
    output = Dense(1)(mlp1)
    
    model = Model(inputs = [m_input, c_input], outputs=output)
    print(output.shape)
    
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mse','mae'])
    model.fit([x_train, c_train], y_train, batch_size=32, epochs=epoch, shuffle=True, validation_data=([x_test, c_test], y_test),
                callbacks=[TensorBoard(log_dir='./tmp/log_cnn_indice')])
    score = model.evaluate([x_test, c_test], y_test)
    print(model.summary())
    print(score)
    return score



# In[96]:

cnn_1att_indice(x_train, x_test, y_train, y_test, train_context, test_context, 100)


# In[97]:

def cnn_att_indice_year(x_train, x_test, y_train, y_test, c_train, c_test, epoch):
    np.random.seed(1)
    m_input = Input(shape=(maxlen, n_features))
    t_x = Permute([2, 1])(m_input)
    print(t_x.shape)
    t_x = Reshape((25, 10))(t_x)
    print('transpose shape:', t_x.shape)
    m_dense = TimeDistributed(Dense(64, activation='relu'))(t_x)
    print(m_dense.shape)
    attention = TimeDistributed(Dense(1, activation='relu'))(m_dense) # try diff act
    print(attention.shape)
    attention = Flatten()(attention)
    print(attention.shape)
    attention = Activation('softmax')(attention) # try different activations
    print(attention.shape)
    attention = RepeatVector(10)(attention)
    print(attention.shape)
    attention = Reshape((10, 25))(attention)
    print('original_shape:', attention.shape)
    
    
    m_dense2 = TimeDistributed(Dense(64, activation='relu'))(m_input)
    attention2 = TimeDistributed(Dense(1, activation='relu'))(m_dense2)
    attention2 = Flatten()(attention2)
    attention2 = Activation('softmax')(attention2)
    attention2 = RepeatVector(25)(attention2)
    attention2 = Permute((2, 1))(attention2)
    print('original_shape2:', attention2.shape)
    
    attention_all = keras.layers.Add()([attention, attention2])

    m_att = merge([m_input, attention_all], mode='mul')
    
    m_model = Sequential()
    m_model.add(Conv1D(filters, input_shape=(maxlen, n_features), kernel_size=(kernel_size,), padding='valid', strides=(1,), activation='relu'))
    m_model.add(GlobalAveragePooling1D())
    
    n_context=4
    c_input = Input(shape=(n_context,))
    
    en_m = m_model(m_att)
    print(en_m.shape)
    meg = keras.layers.concatenate([en_m, c_input])
    
    mlp1 = Dense(10, activation='relu')(meg)
    mlp1 = Dense(5, activation='relu')(mlp1)
    output = Dense(1)(mlp1)
    
    model = Model(inputs = [m_input, c_input], outputs=output)
    print(output.shape)
    
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mse','mae'])
    model.fit([x_train, c_train], y_train, batch_size=32, epochs=epoch, shuffle=True,validation_data=([x_test, c_test], y_test),
                callbacks=[TensorBoard(log_dir='./tmp/log_cnn_yearandindices')])
    score = model.evaluate([x_test, c_test], y_test)
    print(model.summary())
    print(score)
    return score



# In[98]:

cnn_att_indice_year(x_train, x_test, y_train, y_test, train_context, test_context, 100)





