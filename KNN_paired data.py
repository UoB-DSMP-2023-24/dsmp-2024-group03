import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

# 定义文件路径
file_path = 'F:\\Data Science Mini Project\\vdjdb.txt'  # 将 'your_file.txt' 替换为你的文件路径

# 读取文件内容
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取文件的第一行，获取所有的信息变量名
    header = file.readline().strip().split('\t')
    tcr_data = [dict(zip(header, line.strip().split('\t'))) for line in file]
print(header)
cdr3_dict = {}
for row in tcr_data:
    complex_id = row['complex.id']
    cdr3 = row['cdr3']
    # 将相同 complex.id 的 cdr3 拼接起来
    if complex_id in cdr3_dict:
        cdr3_dict[complex_id].append(cdr3)
    else:
        cdr3_dict[complex_id] = [cdr3]
# 假设有一个包含 TCR 序列的 DataFrame
for row in tcr_data:
    complex_id = row['complex.id']
    antigen_epitope = row['antigen.epitope']
    vdjdb_score = row['vdjdb.score']
    # 将相同 complex.id 的 cdr3 拼接起来
    if len(cdr3_dict[complex_id]) == 2:
        cdr3_dict[complex_id].append(antigen_epitope)
        cdr3_dict[complex_id].append(vdjdb_score)
    else:
        continue
cdr3_dict.pop('0')
##删除未配对的TCR
df_cdr3 = pd.DataFrame(cdr3_dict)
df_cdr3_trans = df_cdr3.transpose()
names = ['TRA', 'TRB', 'antigen_epitope', 'vdjdb.score']
df_cdr3_trans.columns = names
print(df_cdr3_trans)
df_clean = df_cdr3_trans[df_cdr3_trans['vdjdb.score'] != '0']
df_clean = df_clean.reset_index(drop=True)
df_clean['TRA_TRB_Combined'] = df_clean["TRA"] + df_clean["TRB"]
print(df_clean)
##----------------接下来编码-------------------------------
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
cdr3_encoded = [[encoding_map[char] for char in sequence] for sequence in df_clean['TRA_TRB_Combined']]
antigen_encoded = [[encoding_map[char] for char in sequence] for sequence in df_clean['antigen_epitope']]
print(df_clean)
##独热码成功编辑，但是矩阵长度不一致
longest_cdr3 = max(df_clean['TRA_TRB_Combined'], key=len)
print("最长的cdr3:", longest_cdr3)
print("最长cdr3的长度:", len(longest_cdr3))
longest_antigen_epitope = max(df_clean['antigen_epitope'], key=len)
print("最长的antigen_epitope:", longest_antigen_epitope)
print("最长antigen_epitope的长度:", len(longest_antigen_epitope))


def padding_sequence(origin, sequence_length):
    padded = np.zeros((sequence_length, 20))
    padded[:len(origin)] = origin
    return padded


cdr3_encoded_padded = [padding_sequence(seq, len(longest_cdr3)) for seq in cdr3_encoded]
antigen_encoded_padded = [padding_sequence(seq, len(longest_antigen_epitope)) for seq in antigen_encoded]
cdr3_encoded_padded_flat = [seq.flatten() for seq in cdr3_encoded_padded]
antigen_encoded_padded_flat = [seq.flatten() for seq in antigen_encoded_padded]
df_clean['cdr3_code'] = cdr3_encoded_padded_flat
df_clean['antigen_code'] = antigen_encoded_padded_flat

##----------------------------建模并运算------------------------------

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 划分特征和目标变量
X = np.array(df_clean['cdr3_code'].tolist())  # cdr_code作为特征
y = np.array(df_clean['antigen_code'].tolist())  # antigen_code作为目标变量
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # 初始化KNN模型
knn_model = KNeighborsRegressor(n_neighbors=1)
multioutput_model = MultiOutputRegressor(knn_model)
multioutput_model.fit(X_train, y_train)
y_pred = multioutput_model.predict(X_test)
def binary_prediction(pred, threshold=0.4):
    return (pred > threshold).astype(int)
binary_pred = binary_prediction(y_pred)
accuracy = accuracy_score(y_test, binary_pred)
print("Accuracy:", accuracy)
