
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Đọc dữ liệu huấn luyện 
training_data = pd.read_csv('UNSW_NB15_training-set (2).csv')
training_data = training_data.drop(['id', 'attack_cat'], axis=1)

#Đọc dữ liệu kiểm tra
testing_data = pd.read_csv('UNSW_NB15_testing-set (2).csv')
testing_data = testing_data.drop(['id', 'attack_cat'], axis=1)

#Xử lý thuộc tính tập huấn luyện
label_encoder = LabelEncoder()
training_data['dur'] = label_encoder.fit_transform(training_data['dur'])
training_data['proto'] = label_encoder.fit_transform(training_data['proto'])
training_data['service'] = label_encoder.fit_transform(training_data['service'])
training_data['state'] = label_encoder.fit_transform(training_data['state'])

#Xử lý thuộc tính tập kiểm tra
testing_data['dur'] = label_encoder.fit_transform(testing_data['dur'])
testing_data['proto'] = label_encoder.fit_transform(testing_data['proto'])
testing_data['service'] = label_encoder.fit_transform(testing_data['service'])
testing_data['state'] = label_encoder.fit_transform(testing_data['state'])
                                         
#Chọn features và label tương ứng
training_features = training_data.drop('label', axis=1)
training_label = training_data[['label']]
training_label = np.ravel(training_label)

#Chọn features và label tương ứng
testing_features = testing_data.drop('label', axis=1)
testing_label = testing_data[['label']]
testing_label = np.ravel(testing_label)

feature_names = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
                 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb',
                 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
                 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
                 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

accuracies = []

for feature in feature_names:
    selected_features = training_features.drop(feature, axis=1), testing_features.drop(feature, axis=1)
    
    # Khởi tạo mô hình RandomForestClassifier
    model = RandomForestClassifier()
    
    # Huấn luyện mô hình
    model.fit(selected_features[0], training_label)
    
    # Dự đoán nhãn cho các mẫu dữ liệu kiểm tra
    y_pred = model.predict(selected_features[1])
    
    # Tính độ chính xác
    accuracy = accuracy_score(testing_label, y_pred)
    accuracies.append(accuracy)

accuracies = [accuracy * 100 for accuracy in accuracies]

for i in range(len(feature_names)):
    print(f"ĐỘ chính xác của mô hình Phát Hiện với thông số '{feature_names[i]}': {accuracies[i]}")

# Vẽ biểu đồ độ chính xác cho từng feature
plt.figure(figsize=(10, 6))
plt.bar(feature_names, accuracies)
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.title('Accuracy for Each Feature')
plt.xticks(rotation=90)
plt.show()



