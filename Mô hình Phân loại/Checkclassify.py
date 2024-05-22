import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#Đọc dữ liệu kiểm tra
testing_data = pd.read_csv('UNSW_NB15_testing-set (2).csv')
testing_data = testing_data.drop(['id'], axis=1)
#testing_data = testing_data[testing_data['label'] == 1]

#Xử lý thuộc tính 
label_encoder = LabelEncoder()
testing_data['dur'] = label_encoder.fit_transform(testing_data['dur'])
testing_data['proto'] = label_encoder.fit_transform(testing_data['proto'])
testing_data['service'] = label_encoder.fit_transform(testing_data['service'])
testing_data['state'] = label_encoder.fit_transform(testing_data['state'])
                                         
#Chọn features và label tương ứng
features = testing_data.drop('attack_cat', axis=1)
label = testing_data[['attack_cat']]

label = np.ravel(label)

#Tải mô hình đã huấn luyện
model = joblib.load('Classify.joblib')
y_pred = model.predict(features)

#Tính độ cxac
accuracy = accuracy_score(label, y_pred)
print('Accuracy:', accuracy)
