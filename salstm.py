# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 00:12:56 2022

@author: USER
"""
'''
import pandas as pd

df=pd.read_excel("tptptkclean2.xlsx",index_col=0,engine="openpyxl")
aa=pd.DataFrame(df)

train = []
label = []


for x in range(len(aa)):
    train.append(aa.iat[x,0])
#print(train)
for y in range(len(aa)):
    label.append(aa.iat[y,1])
#print(label)
    
train = train[:8000]
label = label[:8000]

#打亂資料
import random
x_shuffle=train
y_shuffle=label
z_shuffle = list(zip(x_shuffle, y_shuffle))

random.shuffle(z_shuffle)
x_train, y_label = zip(*z_shuffle)
print(label[:10])
print(y_label[:10])

#label序列化
from keras.utils import np_utils
y_label = np_utils.to_categorical(y_label, 2)
print(y_label[:3])

#將資料分割為訓練資料與測試資料
NUM_TRAIN = int(8000 * 0.8)
train, test = x_train[:NUM_TRAIN], x_train[NUM_TRAIN:]
labels_train, labels_test = y_label[:NUM_TRAIN], y_label[NUM_TRAIN:]

#導入停用詞
stopWords=[]
with open('stopWord.txt', 'r', encoding='utf8') as f:
    stopWords = f.read().split('\n') 
stopWords.append('\n')


#透過jieba分詞工具，分別處理train和test資料
import jieba
sentence=[]
sentence_test=[]


for content in train:
    _sentence=list(jieba.cut(content, cut_all=True))
    sentence.append(_sentence)
for content in test:
    _sentence=list(jieba.cut(content, cut_all=True))
    sentence_test.append(_sentence)
    

#將斷詞分別從train和test資料中移除
remainderWords2 = []
remainderWords_test = []


for content in sentence:
    remainderWords2.append(list(filter(lambda a: a not in stopWords, content)))
for content in sentence_test:
    remainderWords_test.append(list(filter(lambda a: a not in stopWords, content)))
print(train[1])
print(train[:2])
print(remainderWords2[1])
print(remainderWords2[:2])

#建立字典1000字
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
token = Tokenizer(num_words=1000)
token.fit_on_texts(remainderWords2)
#print(token.word_index)

#透過token的texts_to_sequences()方法將train及test轉換為數字list
x_train_seq = token.texts_to_sequences(remainderWords2)
x_test_seq = token.texts_to_sequences(remainderWords_test)
print(x_train_seq[1])

#將序列後的訓練及測試資料長度限制在100
x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)
print(x_train.shape)
#print(x_test.shape)
print(x_train[1])
'''
'''
#測試迴圈(input:a,hidden:b,dropout:c1)
#需要測試將model向右移動至迴圈，抓取資料於model下方
input_layer = []
hidden_layer = []
dropout = []
loss_e = []
val_loss =[]
accuracy_e = []
val_accuracy = []
for a in range(16,129,16):
    for b in range(16,129,16):
        for c in range(20,51,10):
            c1 = c/100
'''   

#建立編譯模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM 
model = Sequential()
model.add(Embedding(1000,16,input_length = 100))#建立詞向量 1000(input_dim)x32(單字長度,特徵)
model.add(Dropout(0.5))
model.add(LSTM(16))#也可以直接將其改成Bi-LSTM,使用Bidirectional。反向可以使用go_backwards=True
model.add(Dropout(0.5))
model.add(Dense(2,activation="sigmoid"))

#model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# 訓練模型
history = model.fit(x_train, labels_train, validation_split=0.2, 
          epochs=10, batch_size=140, verbose=1)

import numpy as np
hv1 = history.history['val_loss']
hv2 = history.history['val_accuracy']
idx1 = np.argmin(hv1)
idx2 = np.argmax(hv2)
val1 = hv1[idx1]
val2 = hv2[idx2]
print("best","val_loss:",idx1+1,val1,"val_accuracy:",idx2+1,val2)

model

'''
#抓取每個參數組合及對應的最佳週期中的loss及準確率  
#並存入excel(依情況跟改檔名)          
            import pandas as pd
            import numpy as np
            hv1 = history.history['val_loss']
            hv2 = history.history['val_accuracy']
            idx1 = np.argmin(hv1)
            idx2 = np.argmax(hv2)
            val1 = hv1[idx1]
            val2 = hv2[idx2]
            print("best",a,b,c1,"val_loss:",idx1+1,val1,"val_accuracy:",idx2+1,val2)
            
            input_layer.append(a)
            hidden_layer.append(b)
            dropout.append(c1)
            loss_e.append(idx1+1)
            val_loss.append(val1)
            accuracy_e.append(idx2+1)
            val_accuracy.append(val2)

model_dict = {'輸入層':input_layer,'隱藏層':hidden_layer,'丟棄層':dropout,
              'loss週期':loss_e,'val_loss':val_loss,
              '準確率週期':accuracy_e,'準確率':val_accuracy}
model_pddict = pd.DataFrame(model_dict)
model_pddict.to_excel('model_data.xlsx',engine='xlsxwriter')
'''

#畫圖
import matplotlib.pyplot as plt
 
def plot(history_dict,keys,title=None,xyLabel=[],ylim=(),size=()):
    lineType = ('-','--','.',':')
    if len(ylim)==2 : plt.ylim(*ylim)
    if len(size)==2 : plt.gcf().set_size_inches(*size)
    epochs = range(1,len(history_dict[keys[0]])+1)
    for i in range(len(keys)):
        plt.plot(epochs,history_dict[keys[i]],lineType[i])
    if title:
        plt.title(title)
    if len(xyLabel)==2:
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys,loc='best')
    plt.show()

plt.subplot(1,2,1)
plot(history.history,
       ('loss','val_loss'),
       'Training and Validation Loss',
       ('Epoch','loss'))
plt.subplot(1,2,2)
plot(history.history,
       ('accuracy','val_accuracy'),
       'Training and Validation accuracy',
       ('Epoch','accuracy'))


model.save('lstm_SA.h5')
scores = model.evaluate(x_test, labels_test, verbose=1)
scores[1]

predict= model.predict(x_test)
print('第2筆資料之預測機率:',predict[1])

predict_class = model.predict_classes(x_test)
print('前10筆預測標籤:',predict_class[:10])
print('前10筆正確標籤:',predict_class[:10])























