from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

##------------------读取文件----------------------------------------------------
# 定义文件路径
file_path = 'F:\\Data Science Mini Project\\vdjdb.txt'  # 将 'your_file.txt' 替换为你的文件路径

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取文件的第一行，获取所有的信息变量名
    header = file.readline().strip().split('\t')
    tcr_data = [dict(zip(header, line.strip().split('\t'))) for line in file]
print(header)
##--------------------------------------------------------------------


# --------------清洗第一步，提取所需属性-----------------------------------
selected_data = [{'cdr3': entry['cdr3'],
                  'antigen.epitope': entry['antigen.epitope'],
                  'vdjdb.score': entry['vdjdb.score']}
                 for entry in tcr_data]
##------------------------------------------------------------------------


# ---清洗第二步，转化为数据集，并删去重复元素,同时删除可信度低的行-----------------------
df_raw = pd.DataFrame(selected_data)
df_clean = df_raw[df_raw['vdjdb.score'] != '0']
#df_clean = df_clean.drop_duplicates()
df_clean = df_clean.reset_index(drop=True)
print(df_clean)
##-------------------------------------------------------------------------------


##---编码-----------------------------------------------------------------------
amino_acids = set(''.join(df_clean['cdr3']))
encoding_map = {'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
cdr3_encoded = [[encoding_map[char] for char in sequence] for sequence in df_clean['cdr3']]
antigen_encoded = [[encoding_map[char] for char in sequence] for sequence in df_clean['antigen.epitope']]
# 打印带有编码的数据集
print(df_clean)
##独热码成功编辑，但是矩阵长度不一致
longest_cdr3 = max(df_clean['cdr3'], key=len)
print("最长的cdr3:", longest_cdr3)
print("最长cdr3的长度:", len(longest_cdr3))
longest_antigen_epitope = max(df_clean['antigen.epitope'], key=len)
print("最长的antigen_epitope:", longest_antigen_epitope)
print("最长antigen_epitope的长度:", len(longest_antigen_epitope))


def padding_sequence(origin, sequence_length):
    padded = np.zeros((sequence_length, 20))
    padded[:len(origin)] = origin
    return padded


cdr3_encoded_padded = [padding_sequence(seq,len(longest_cdr3)) for seq in cdr3_encoded]
antigen_encoded_padded=[padding_sequence(seq,len(longest_antigen_epitope)) for seq in antigen_encoded]
cdr3_encoded_padded_flat=[seq.flatten()  for seq in cdr3_encoded_padded ]
antigen_encoded_padded_flat=[seq.flatten()  for seq in antigen_encoded_padded ]
df_clean['cdr3_code'] = cdr3_encoded_padded_flat
df_clean['antigen_code'] = antigen_encoded_padded_flat

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 划分特征和目标变量
X = np.array(df_clean['cdr3_code'].tolist())  # cdr_code作为特征
y = np.array(df_clean['antigen_code'].tolist())  # antigen_code作为目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 初始化MLP模型
mlp_model = MLPRegressor(hidden_layer_sizes=(2048,1024,512), activation='relu', solver='adam', max_iter=100, random_state=42)
# 训练模型
mlp_model.fit(X_train, y_train)

# 在测试集上做预测
y_pred = mlp_model.predict(X_test)

# 根据阈值将预测值转换为二元输出
def binary_prediction(pred, threshold=0.4):
    return (pred > threshold).astype(int)
# 在测试集上进行预测
y_pred = mlp_model.predict(X_test)

# 将预测值转换为二元输出
binary_pred = binary_prediction(y_pred)

# 计算精度
accuracy = accuracy_score(y_test, binary_pred)
print("Accuracy:", accuracy)
##----------------------------
