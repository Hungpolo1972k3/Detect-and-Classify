import joblib
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
model1= joblib.load('detect.joblib')
model2= joblib.load('Classify.joblib')

#Đọc file csv 
new_data = pd.read_csv("test7.csv")
##Xử lý thuộc tính 
label_encoder = LabelEncoder()
new_data['proto'] = label_encoder.fit_transform(new_data['proto'])
new_data['service'] = label_encoder.fit_transform(new_data['service'])
new_data['state'] = label_encoder.fit_transform(new_data['state'])

#Sử dụng model 1 để gán nhãn cho dữ liệu đầu vào
predictions1 = model1.predict(new_data)

new_data['label'] = [1] # Giả sử bị tấn công
#Sử dụng model 2 để gán nhãn cho Dataframe nếu giả sử bị tấn công
predictions = model2.predict(new_data)

#Kiểm tra
print('File csv đầu vào thứ 7')
for i, prediction in enumerate(predictions1):
    if prediction == 1:
        print("Có tấn công : ",predictions)
    else:
        print("Không bị tấn công")
