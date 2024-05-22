import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Đọc dữ liệu huấn luyện 
training_data = pd.read_csv('UNSW_NB15_training-set (2).csv')
training_data = training_data.drop(['id'], axis=1)

#Đọc dữ liệu kiểm tra
testing_data = pd.read_csv('UNSW_NB15_testing-set (2).csv')
testing_data = testing_data.drop(['id'], axis=1)

#Xử lý thuộc tính tệp huấn luyện 
label_encoder = LabelEncoder()
training_data['dur'] = label_encoder.fit_transform(training_data['dur'])
training_data['proto'] = label_encoder.fit_transform(training_data['proto'])
training_data['service'] = label_encoder.fit_transform(training_data['service'])
training_data['state'] = label_encoder.fit_transform(training_data['state'])
                                        
#Xử lý thuộc tính tệp kiểm tra
testing_data['dur'] = label_encoder.fit_transform(testing_data['dur'])
testing_data['proto'] = label_encoder.fit_transform(testing_data['proto'])
testing_data['service'] = label_encoder.fit_transform(testing_data['service'])
testing_data['state'] = label_encoder.fit_transform(testing_data['state'])

#Chọn features và label tương ứng tệp huấn luyện
training_features = training_data.drop('attack_cat', axis=1)
training_label = training_data[['attack_cat']]
training_label = np.ravel(training_label)

#Chọn features và label tương ứng tệp kiểm tra
testing_features = testing_data.drop('attack_cat', axis=1)
testing_label = testing_data[['attack_cat']]
testing_label = np.ravel(testing_label)

#Khởi tạo mô hình RandomForestClassifier
model = RandomForestClassifier(n_estimators = 150, max_depth = 17 )

#Huấn luyện mô hình   
model.fit(training_features, training_label)

#Dự đoán nhãn cho các mẫu dữ liệu kiểm tra
y_pred = model.predict(testing_features)

#Tính độ cxac
accuracy = accuracy_score(testing_label, y_pred)
print('Accuracy:', accuracy)

#Lưu trữ mô hình đã được huấn luyện
joblib.dump(model, 'Classify.joblib')

