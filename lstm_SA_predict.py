# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:53:23 2022

@author: USER
"""


from tensorflow.keras.models import load_model
salstm_model = load_model('lstm_SA.h5')



scores = model.evaluate(x_test, labels_test, verbose=1)
scores[1]

predict= model.predict(x_test)
print('第2筆資料之預測機率:',predict[1])

predict_class = model.predict_classes(x_test)
print('前10筆預測標籤:',predict_class[:20])
print('前10筆正確標籤:',predict_class[:20])