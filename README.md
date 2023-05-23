# LSTM_Sentiment-Analysis_zh-tw

此項目使用 [selenium-crawler-googlemap](https://github.com/cyyW/selenium-crawler-googlemap) 中爬取到的飲料店資料，訓練前將留言內容用 Jieba-tw 進行段詞及去除[停用字](https://github.com/cyyW/LSTM_Sentiment-Analysis_zh-tw/blob/main/stopWord.txt)之處理，並將留言作為樣本、星數作為標籤進行 LSTM 的情緒分析之訓練並使其預測未知資料為正面或負面之情緒。

* 訓練檔 : [salstm.py](https://github.com/cyyW/LSTM_Sentiment-Analysis_zh-tw/blob/main/salstm.py)
* 預測檔 : [lstm_SA_predict.py](https://github.com/cyyW/LSTM_Sentiment-Analysis_zh-tw/blob/main/lstm_SA_predict.py)

## 環境
* python `3.9.15`
* 虛擬環境使用 `conda`
* jieba-tw : https://github.com/APCLab/jieba-tw

## 提示
* 更詳細內容可參考[工管四丙 lstm 報告.docx](https://github.com/cyyW/LSTM_Sentiment-Analysis_zh-tw/blob/main/%E5%B7%A5%E7%AE%A1%E5%9B%9B%E4%B8%99%20lstm%20%E5%A0%B1%E5%91%8A.docx)。
* 可在 [model_data](https://github.com/cyyW/LSTM_Sentiment-Analysis_zh-tw/blob/main/model_data.xlsx)中查看訓練實驗過程紀錄的相關數據。
