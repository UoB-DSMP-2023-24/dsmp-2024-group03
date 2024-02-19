import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
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
                  'gene':entry['gene'],
                  'v.segm':entry['v.segm'],
                  'j.segm':entry['j.segm'],
                  'antigen.epitope': entry['antigen.epitope'],
                  'antigen.gene':entry['antigen.gene'],
                  'antigen.species':entry['antigen.species'],
                  'vdjdb.score': entry['vdjdb.score'],
                  'mhc.a': entry['mhc.a'],
                  'mhc.b': entry['mhc.b'],
                  'mhc.class':entry['mhc.class']
                  }
                 for entry in tcr_data]
##------------------------------------------------------------------------


# ---清洗第二步，转化为数据集，并删去重复元素,同时删除可信度低的行-----------------------
df_raw = pd.DataFrame(selected_data)
df_clean = df_raw[df_raw['vdjdb.score'] != '0']
df_clean.drop(columns=['vdjdb.score'], inplace=True)
df_clean = df_clean.reset_index(drop=True)
print(df_clean)


##----------------相关分析------------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

n_features = len(df_clean.columns)
corr_matrix = np.zeros((n_features, n_features))
p_value_matrix = np.zeros((n_features, n_features))

# 循环计算所有特征两两之间的Spearman相关系数和p值
for i in range(n_features):
    for j in range(n_features):
        corr, p_value = spearmanr(df_clean.iloc[:, i], df_clean.iloc[:, j])
        column_name = df_clean.columns[i]
        corr_matrix[i, j] = corr
        p_value_matrix[i, j] = p_value
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Spearman Correlation Heatmap')
plt.show()
##----------------------------------------------------------------------------

