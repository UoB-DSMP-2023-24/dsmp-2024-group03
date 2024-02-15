from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.reset_index(drop=True)
print(df_clean)
##------------------------------------------------------------------

##---------编码，使用词袋编码——----------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
# # 获取cdr3和antigen.epitope的序列列表
#cdr_sequences = df_clean['cdr3'].tolist()
#antigen_sequences = df_clean['antigen.epitope'].tolist()
# # 合并cdr和antigen的序列列表
#sequences = cdr_sequences + antigen_sequences
# # 初始化词袋模型
#vectorizer = CountVectorizer(analyzer='char', lowercase=False)  # 将单词拆分为字母，不转换为小写
# # 对序列进行拟合和转换
#X = vectorizer.fit_transform(sequences)
# # 获取编码后的特征矩阵
#cdr_code_bow = X[:len(cdr_sequences)]
#antigen_code_bow = X[len(cdr_sequences):]
# # 将特征矩阵转换为DataFrame
#cdr_code_bow_df = pd.DataFrame(cdr_code_bow.toarray(), columns=vectorizer.get_feature_names_out())
#cdr_code_bow_df['cdr3_code'] = cdr_code_bow_df.apply(lambda row: row[0:].tolist(), axis=1)
#cdr_code_bow_df = cdr_code_bow_df[['cdr3_code']]
#antigen_code_bow_df = pd.DataFrame(antigen_code_bow.toarray(), columns=vectorizer.get_feature_names_out())
#antigen_code_bow_df['antigen_code'] = antigen_code_bow_df.apply(lambda row: row[0:].tolist(), axis=1)
#antigen_code_bow_df = antigen_code_bow_df[['antigen_code']]
# # 将编码后的数据与原始数据合并
#df_clean = pd.concat([df_clean, cdr_code_bow_df, antigen_code_bow_df], axis=1)
# # 删除原始的序列数据，只保留编码后的数据
#print(df_clean)
##------------------------------------------------------------------



##-----------------------编码，现使用独热编码-------------------------------------------------------
amino_acids = set(''.join(df_clean['cdr3']))
amino_acid_to_index = {amino_acid: i for i, amino_acid in enumerate(amino_acids)}
print(amino_acid_to_index)
for amino_acid in amino_acid_to_index:
    df_clean[amino_acid] = df_clean['cdr3'].apply(lambda x: 1 if amino_acid in x else 0)

df_clean['cdr3_code'] = df_clean.apply(lambda row: row[3:].tolist(), axis=1)
df_clean = df_clean[['cdr3', 'antigen.epitope', 'vdjdb.score', 'cdr3_code']]
for amino_acid in amino_acid_to_index:
    df_clean[amino_acid] = df_clean['antigen.epitope'].apply(lambda x: 1 if amino_acid in x else 0)
df_clean['antigen_code'] = df_clean.apply(lambda row: row[4:].tolist(), axis=1)
df_clean = df_clean[['cdr3', 'antigen.epitope', 'vdjdb.score', 'cdr3_code', 'antigen_code']]

##--------------编码过程-------------------------------------------------------------------------


##------------建模预测----------
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 划分特征和目标变量
X = np.array(df_clean['cdr3_code'].tolist())  # cdr_code作为特征
y = np.array(df_clean['antigen_code'].tolist())  # antigen_code作为目标变量

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch的Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# 构建神经网络模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# 初始化模型
input_size = X_train_tensor.shape[1]
hidden_size = 128
output_size = y_train_tensor.shape[1]
model = Net(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 在测试集上评估模型
with torch.no_grad():
    outputs = model(X_test_tensor)
    mse = criterion(outputs, y_test_tensor)
    print("Mean Squared Error on Test Set:", mse.item())
    # 将模型输出的Tensor转换为numpy数组
    predicted_sequences = outputs.numpy().tolist()
    true_sequences = y_test_tensor.numpy().tolist()
    print(predicted_sequences[2])
    print(true_sequences[2])
    # 计算准确率



##----------------------------

def ED(str_1, str_2):
    m = len(str_1)
    n = len(str_2)
    # Initializes the dynamic programming matrix with sizes m+1 and n+1 respectively
    Distance = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(n + 1):
        Distance[0][i] = i
    #
    for j in range(m + 1):
        Distance[j][0] = j
    # Initialize the first row and column of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            distance_delete = Distance[i - 1][j] + 1
            distance_add = Distance[i][j - 1] + 1
            if str_1[i - 1] == str_2[j - 1]:
                distance_change = Distance[i - 1][j - 1]
            else:
                distance_change = Distance[i - 1][j - 1] + 1
            Distance[i][j] = min(distance_delete, distance_add, distance_change)
    # Count the items from bottom to top
    return Distance[m][n]


# define our own similarity function
def similarity(x, scale):
    return 1 / (1 + scale * x)
