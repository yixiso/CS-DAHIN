import json
import pandas as pd
import pickle
import numpy as np
import scipy.sparse as sp
import itertools
import random
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

# 读取 JSON 文件内容
with open('sports_review_data.json', 'r') as file:
    data = json.load(file)

with open('sportsdata1.json', 'r') as features_file:
    features_data = json.load(features_file)


#遍历稀疏矩阵
def sum_and_divide_sparse_matrix(matrix):
    values = matrix.data  # 获取稀疏矩阵中的非零元素值
    total_sum = sum(values)  # 计算非零元素的总和
    average = total_sum  # 将总和除以2得到平均值
    return average

# 提取不重复的 asin 和 reviewerid
reviewerID_set = set()

for item in data:
    # asin_set.add(item['asin'])
    reviewerID_set.add(item['reviewerID'])

#建立索引
# asin_index = {asin: index for index, asin in enumerate(asin_set)}
reviewerID_index = {reviewerID: index for index, reviewerID in enumerate(reviewerID_set)}

# 计算种类数
# asin_count = len(asin_set)

# asin_count = len(asin_set)
reviewerid_count = len(reviewerID_set)

print(reviewerid_count)

features_data_labels = []
features_data_f = []


#brand和asin的键值对
brand_dict = {}
asin_set = set()
asin_index = {}
brand_index = {}
brand_set = set()
label_set = set()
brands = []
#全部特征标签
for line in features_data:
    asin = line['asin']
    brand = line['brand']
    label = line['category']
    feature = line['feature']
    if asin not in asin_set:
        asin_index[asin] = len(features_data_labels)
        if label == 'Sports & Fitness':
            features_data_labels.append('0')
        if label == 'Fan Shop':
            features_data_labels.append('2')
        if label == 'Outdoor Recreation':
            features_data_labels.append('1')
        # features_data_labels.append(line['category'])
        line['feature'] = feature.replace('|', ',')
        features_data_f.append(line['feature'])
        asin_set.add(asin)
        brand_set.add(brand)
        label_set.add(label)
        brands.append(brand)  # 将 'brand' 添加到列表

#建立brands索引
brand_list = list(brand_set)
brand_to_index = {brand: index for index, brand in enumerate(set(brands))}
asin_count = len(asin_set)
brand_count = len(brand_set)
label_count = len(label_set)
asin_brand = {}
for line in features_data:
    asin_id = asin_index[line['asin']]
    for i in range(len(brand_list)):
        if line['brand'] == brand_list[i]:
            asin_brand[asin_id] = i
            break



print("label：",label_count)
print("asin：",asin_count)

#处理特征
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vectorizer = CountVectorizer(min_df=58, stop_words=sklearn_stopwords.union(set(nltk_stopwords.words('english'))), tokenizer=LemmaTokenizer())
# print(features_data_f)
author_X = vectorizer.fit_transform(features_data_f)
# vectorizer = CountVectorizer(min_df=1,stop_words=None,analyzer='char')
# author_X = vectorizer.fit_transform(features_data_f)
print("author_X 的形状：", author_X.shape)
print("author_X 的存储格式：", author_X.getformat())  

snaps = []
snap_raw = []

# 将数据按年份分组
snapshots = {}

iusum = 0
ibsum = 0
iuisum = 0
ibisum = 0

for item in data:
    review_time = item.get('reviewTime')
    if review_time:
        year = int(review_time.split(',')[-1].strip())
        group = (year - 2001) // 3
        year_range = f"{2001 + group*3}-{2001 + group*3 + 2}"
        if year_range not in snapshots:
            snapshots[year_range] = []
        snapshots[year_range].append(item)

for year_range, items in snapshots.items():
    # adjacency_matrix_iui = coo_matrix((asin_count, asin_count), dtype=np.float64)
    # adjacency_matrix_iu = coo_matrix((asin_count, reviewerid_count), dtype=np.float64)
    #创建空列表用于存储非零元素的行索引、列索引和对应值
    row_indices = []
    col_indices = []
    col_indeces_brand = []
    values = []
    values1 = []
    valid_nodes = []
    # 首先取得当前快照内的iu邻接矩阵
    for i in items:
        asin_id = asin_index[i['asin']]
        reviewer_id = reviewerID_index[i['reviewerID']]
        row_indices.append(asin_id)
        col_indices.append(reviewer_id)
        values.append(1)
    # 去除重复的边
    unique_edges = set(zip(row_indices, col_indices))
    unique_edges_brand = set(zip(row_indices, col_indeces_brand))

    # 重新构建稀疏矩阵
    row_indices = [row for row, col in unique_edges]
    col_indices = [col for row, col in unique_edges]
    values = [values[i] for i in range(len(unique_edges))]

   

    adjacency_matrix_iu = coo_matrix((values, (row_indices, col_indices)), shape=(asin_count, reviewerid_count), dtype=np.int64)
    adjacency_matrix_iui = np.dot(adjacency_matrix_iu, np.transpose(adjacency_matrix_iu))
    valid_nodes = np.nonzero(np.sum(adjacency_matrix_iui, axis=1))[0]
    validnode = valid_nodes
    adjacency_matrix_valid = adjacency_matrix_iui[validnode][:, validnode]

    iusum +=sum_and_divide_sparse_matrix(adjacency_matrix_iu)
    iuisum +=sum_and_divide_sparse_matrix(adjacency_matrix_iui)


    #ibi
    row = np.array(validnode)
    col = np.array([asin_brand[asin_id] for asin_id in validnode])
    data = np.ones(len(validnode))
    adjacency_matrix_ib = coo_matrix((data, (row, col)),shape=(asin_count, brand_count), dtype=np.int64)
    adjacency_matrix_ibi = np.dot(adjacency_matrix_ib, np.transpose(adjacency_matrix_ib))

    adjacency_matrix_valid_ibi = adjacency_matrix_ibi[validnode][:, validnode]

    ibsum +=sum_and_divide_sparse_matrix(adjacency_matrix_ib)
    ibisum +=sum_and_divide_sparse_matrix(adjacency_matrix_ibi)

    #获取特征矩阵
    valid_nodes_len = len(valid_nodes)
    valid_nodes_features = [''] * len(valid_nodes)
    valid_features_X = author_X[validnode]
    # for i, node_index in enumerate(valid_nodes):
    #     # print("all_labels:",all_labels_str[node_index])
    #     valid_nodes_features[i] = features_data_labels[node_index]
    #     # print("valid:",valid_nodes_features[i])

    #将标签处理为特征
    # vectorizer = CountVectorizer(min_df=1,stop_words=None,analyzer='char')
    # author_X = vectorizer.fit_transform(valid_nodes_features)
    # print("author_X 的形状：", author_X.shape)
    # print("author_X 的存储格式：", author_X.getformat())     

    # #转换稀疏矩阵   
    # sparse_matrix_v = csr_matrix(adjacency_matrix_valid)
    # sparse_matrix_raw = csr_matrix(adjacency_matrix_iui)
    print('ansap')
    snap = {
        "adjMs":{
        'adjacency_matrix_iui': adjacency_matrix_valid,
        'adjacency_matrix_ibi':adjacency_matrix_valid_ibi,
        },
        # "features":valid_nodes_features,
        "features":valid_features_X,
        "valid_nodes":validnode
    }
    # print("特征：",valid_features_X)
    snaps.append(snap)

    snapraw = {
        'adjacency_matrix_iui': adjacency_matrix_iui,
        'adjacency_matrix_ibi':adjacency_matrix_ibi
    }
    snap_raw.append(snapraw)


#随机抽取训练集测试集等
# train_size = 1000
# valid_size = 200
# test_size = 1000

# indices = list(range(asin_count))

# train_indices = random.sample(indices, train_size)

# remaining_indices = set(indices) - set(train_indices)
# valid_indices = random.sample(list(remaining_indices), valid_size)

# test_indices = random.sample(list(remaining_indices - set(valid_indices)), test_size)
# print(len(all_labels))


data = {
    'labels':features_data_labels,
    'snaps':snaps,
    "snaps_raw":snap_raw,
    'nodes_num':asin_count,
    'target_type':0,
    "type":['item','user','brand'],
    "metapath":["item-user-item","item-brand-item"],
    "time—scale":"3year",
    # "train_idx":train_indices,
    # "val_idx":valid_indices,
    # "test_idx":test_indices
}
print(brand_count)

with open('sportsdata.pkl', 'wb') as f:
    pickle.dump(data, f)

    # 读取pkl文件
with open("sportsdata.pkl", "rb") as f:
    all_data = pickle.load(f)

print(all_data)
print("用户数：",reviewerid_count,"物品数：",asin_count,"品牌数：",brand_count)
print("iui边数：",iuisum,"ibi边数：",ibisum,"iu边数：",iusum,"ib边数:",ibsum)