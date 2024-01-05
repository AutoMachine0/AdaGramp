import faiss
import torch
import numpy as np
import pandas as pd
from scipy.io import loadmat
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################### NEGATIVE SAMPLE FUNCTIONS################################################
def negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, sample_type, proportion, epsilon):
    
    # training set negative samples
    neg_train_x, neg_train_y = generate_negative_samples(train_x, sample_type, proportion, epsilon)
    # validation set negative samples
    neg_val_x, neg_val_y = generate_negative_samples(val_x, sample_type, proportion, epsilon)
    
    # concat data
    # 正负样本x,y数据集拼接
    x = np.vstack((train_x, neg_train_x, val_x, neg_val_x, test_x))
    y = np.hstack((train_y, neg_train_y, val_y, neg_val_y, test_y))

    # 为构造图 transductive graph learning训练，验证，测试做准备mask
    # all training set
    train_mask = np.hstack((np.ones(len(train_x)), np.ones(len(neg_train_x)),
                            np.zeros(len(val_x)), np.zeros(len(neg_val_x)),
                            np.zeros(len(test_x))))
    # all validation set
    val_mask = np.hstack((np.zeros(len(train_x)), np.zeros(len(neg_train_x)),
                          np.ones(len(val_x)), np.ones(len(neg_val_x)),
                          np.zeros(len(test_x))))
    # all test set
    test_mask = np.hstack((np.zeros(len(train_x)), np.zeros(len(neg_train_x)),
                           np.zeros(len(val_x)), np.zeros(len(neg_val_x)),
                           np.ones(len(test_x))))
    # 为正常点构造邻居节点做准备
    # normal training points
    neighbor_mask = np.hstack((np.ones(len(train_x)), np.zeros(len(neg_train_x)), 
                               np.zeros(len(val_x)), np.zeros(len(neg_val_x)),
                               np.zeros(len(test_x))))
    
    # find k nearest neighbours (idx) and their distances (dist) to each points in x within neighbour_mask==1
    dist, idx = find_neighbors(x, y, neighbor_mask, k)

    return x.astype('float32'), y.astype('float32'), neighbor_mask.astype('float32'), train_mask.astype('float32'), val_mask.astype('float32'), test_mask.astype('float32'), dist, idx

# loading negative samples
def generate_negative_samples(x, sample_type, proportion, epsilon):

    # 获取训练集样本数量，特征维度
    n_samples = int(proportion*(len(x)))
    n_dim = x.shape[-1]

    # 构造随机矩阵，维度(n_samples, n_dim),小于0.3的元素为True,大于等于0.3的元素为False 为扰动负采样构造x做准备.
    randmat = np.random.rand(n_samples, n_dim) < 0.3
    # uniform samples均匀采样生成negative x　矩阵,epsilon是超参数,
    rand_unif = (epsilon*(1-2*np.random.rand(n_samples, n_dim)))
    #  subspace perturbation samples / np.tile(x,(proportion, 1))：以ｘ模板按比例(proportion, 1)生成一样的矩阵比例必须为正整数
    rand_sub = np.tile(x, (proportion, 1)) + randmat*(epsilon*np.random.randn(n_samples, n_dim))
    
    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    if sample_type == 'SUBSPACE':
        neg_x = rand_sub
    if sample_type == 'MIXED':
        # randomly sample from uniform and gaussian negative samples
        #　将均匀负采样x与扰动负采样x按行拼接
        neg_x = np.concatenate((rand_unif, rand_sub), 0)
        # 从拼接后的neg_x矩阵中随机有放回的取出与train_x样本数量一致的neg_x
        neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size=n_samples)]
    # 构造负样本标签
    neg_y = np.ones(len(neg_x))
    
    return neg_x.astype('float32'), neg_y.astype('float32')

################################### GRAPH FUNCTIONS ###############################################     
# find the k nearest neighbours of all x points out of the neighbour candidates
def find_neighbors(x, y, neighbor_mask, k):

    # nearest neighbour object
    # 使用faiss暴力计所有正常样本点相互之间的L2距离(欧式距离)
    index = faiss.IndexFlatL2(x.shape[-1]) #　初始化输入计算样本点的维度
    # add nearest neighbour candidates
    # 输入需要计算距离的矩阵x
    index.add(x[neighbor_mask==1]) # 进行L2距离计算的比较对象，训练中正常点向量

    # distances and idx of neighbour points for the neighbour candidates (k+1 as the first one will be the point itself)
    # 找到x矩阵中每个样本最L2距离小的前ｋ个点，返回其L2距离矩阵与在x中索引矩阵
    dist_train, idx_train = index.search(x[neighbor_mask==1], k=k+1)
    # remove 1st nearest neighbours to remove self loops
    # 移除自身节点L2距离与x中索引矩阵中的索引元素
    dist_train, idx_train = dist_train[:, 1:], idx_train[:, 1:]

    # distances and idx of neighbour points for the non-neighbour candidates
    # 为x样本矩阵中非训练集中正常样本点的其余所有的样本点基于x[neighbor_mask==1]为比较比较对象构造最近距离与所索引属性
    # idx_test里的索引是x[neighbor_mask==1]矩阵中的离此点最近的10个点的索引号
    # 基于训练中正常样本点给其余所有点构造距离与邻居索引属性！！！
    dist_test, idx_test = index.search(x[neighbor_mask==0], k=k)

    # 构造graph embedding　与　graph index属性
    # 以训练集中正常样本为邻居候选项为整个数据集构图！！！
    dist = np.vstack((dist_train, dist_test))
    idx = np.vstack((idx_train, idx_test))
    
    return dist, idx

# create graph object out of x, y, distances and indices of neighbours
def build_graph(x, y, dist, idx, k):
    
    # array like [0,0,0,0,0,1,1,1,1,1,...,n,n,n,n,n] for k = 5 (i.e. edges sources)
    # 构建　source index, source node是中心节点(整个图的节点编号)
    idx_source = np.repeat(np.arange(len(x)), dist.shape[-1]).astype('int32')
    idx_source = np.expand_dims(idx_source, axis=0)

    # edge targets, i.e. the nearest k neighbours of point 0, 1,..., n
    # 构建　target index, target node是邻居节点编号(邻居节点都是训练集中的正常点)
    idx_target = idx.flatten()
    idx_target = np.expand_dims(idx_target, axis=0).astype('int32')
    
    #　stack source and target indices
    #　构建PYG需要的edge_inex数据结构
    idx = np.vstack((idx_source, idx_target))

    # edge weights　使用距离构造边特征
    # dist.flatten()：将矩阵dist shape=[node_num,k]拉平得到一维数组attr shape=[node_num*k,]
    attr = dist.flatten()
    attr = np.sqrt(attr)
    attr = np.expand_dims(attr, axis=1)
    
    # into tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    idx = torch.tensor(idx, dtype=torch.long)
    attr = torch.tensor(attr, dtype=torch.float32)

    # build PyTorch geometric Data object　构造pyg图数据结构
    data = Data(x=x, edge_index=idx, edge_attr=attr, y=y)
    data.k = k

    return data

########################################## DATASET FUNCTIONS ####################################
# split training data into train set and validation set
def split_data(all_train_x, all_train_y, all_test_x, all_test_y):
    # np.random.seed(seed)

    # 随机不放回从训练集中取15%的idx构造val_idx
    val_idx = np.random.choice(np.arange(len(all_train_x)), size=int(0.15*len(all_train_x)), replace=False)
    # 构造mask标志位从all_train_x,all_train_y中分离train_x,train_y,val_x,val_y
    val_mask = np.zeros(len(all_train_x))
    val_mask[val_idx] = 1

    val_x = all_train_x[val_mask == 1]
    val_y = all_train_y[val_mask == 1]

    train_x = all_train_x[val_mask == 0]
    train_y = all_train_y[val_mask == 0]
    
    scaler = MinMaxScaler()
    # 以训练集中的min x为中心,(max x - min x)为缩放尺度给训练,验证,测试集进行归一化
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
   
    if all_test_x is None:
        test_x = val_x
        test_y = val_y
    
    test_x = scaler.transform(all_test_x)
    test_y = all_test_y

    return train_x.astype('float32'), train_y.astype('float32'), val_x.astype('float32'), val_y.astype('float32'),  test_x.astype('float32'), test_y.astype('float32')

# load data
def load_dataset(dataset):
    # np.random.seed(seed)
    
    if dataset == 'MI-V':
        df = pd.read_csv("data/MI/experiment_01.csv")
        for i in ['02', '03', '11', '12', '13', '14', '15', '17', '18']:
            data = pd.read_csv("data/MI/experiment_%s.csv"%i)
            df = df.append(data, ignore_index=True)
        normal_idx = np.ones(len(df))
        for i in ['06', '08', '09', '10']:
            data = pd.read_csv("data/MI/experiment_%s.csv"%i)
            df = df.append(data, ignore_index=True)
            normal_idx = np.append(normal_idx, np.zeros(len(data)))
        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'], axis=1), machining_process_one_hot],axis=1)
        data = df.to_numpy()
        idx = np.unique(data,axis=0, return_index=True)[1]
        data = data[idx]
        normal_idx = normal_idx[idx]
        normal_data = data[normal_idx == 1]
        anomaly_data = data[normal_idx == 0]
        test_idx = np.random.choice(np.arange(0, len(normal_data)), len(anomaly_data), replace=False)
        train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data, normal_data[test_idx]))
        test_y = np.concatenate((np.ones(len(anomaly_data)), np.zeros(len(test_idx))))
        
    elif dataset == 'MI-F':

        df = pd.read_csv("data/mi/experiment_01.csv")
        for i in ['02', '03', '06', '08', '09', '10', '11', '12', '13', '14', '15', '17', '18']:
            data = pd.read_csv("data/mi/experiment_%s.csv"%i)
            df = df.append(data, ignore_index=True)
        normal_idx = np.ones(len(df))

        for i in ['04', '05', '07', '16']: 
            data = pd.read_csv("data/mi/experiment_%s.csv"%i)
            df = df.append(data, ignore_index=True)
            normal_idx = np.append(normal_idx, np.zeros(len(data)))

        machining_process_one_hot = pd.get_dummies(df['Machining_Process'])
        df = pd.concat([df.drop(['Machining_Process'], axis=1), machining_process_one_hot], axis=1)
        data = df.to_numpy()
        idx = np.unique(data, axis=0, return_index=True)[1]
        data = data[idx]
        normal_idx = normal_idx[idx]
        normal_data = data[normal_idx == 1]
        anomaly_data = data[normal_idx == 0]
        test_idx = np.random.choice(np.arange(0, len(normal_data)), len(anomaly_data), replace=False)
        train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = np.zeros(len(train_x))
        test_x = np.concatenate((anomaly_data, normal_data[test_idx]))
        test_y = np.concatenate((np.ones(len(anomaly_data)), np.zeros(len(test_idx))))
        
    elif dataset in ['OPTDIGITS', 'PENDIGITS', 'SHUTTLE', "ANNTHYROID"]:
        if dataset == 'SHUTTLE':
            data = loadmat("data/SHUTTLE/shuttle.mat")
        elif dataset == 'OPTDIGITS':
            data = loadmat("data/OPTDIGITS/optdigits.mat")
        elif dataset == 'PENDIGITS':
            data = loadmat('data/PENDIGITS/pendigits.mat')
        elif dataset == 'ANNTHYROID':
            data = loadmat('data/ANNTHYROID/annthyroid.mat')

        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        normal_data = data[label == 0]
        normal_label = label[label == 0]
        anom_data = data[label == 1]
        anom_label = label[label == 1]
        test_idx = np.random.choice(np.arange(0, len(normal_data)), len(anom_data), replace=False)
        train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = np.concatenate((normal_data[test_idx], anom_data))
        test_y = np.concatenate((normal_label[test_idx], anom_label))
        
    elif dataset in ['HRSS']:
        if dataset == 'HRSS':
            data = pd.read_csv('data/HRSS/HRSS.csv').to_numpy()

        # 获取 标签y向量
        label = data[:, -1].astype('float32').squeeze()
        # 获取　特征x特征矩阵
        data = data[:, :-1].astype('float32')
        # 获取正常点特征x与标签y
        normal_data = data[label == 0]
        normal_label = label[label == 0]
        # 获取异常点特征x与标签y
        anom_data = data[label == 1]
        anom_label = label[label == 1]

        # 从正常点中不放回地随机取与异常点相同数量的点id作为test_idx
        test_idx = np.random.choice(np.arange(0, len(normal_data)), len(anom_data), replace=False)
        # 从正常点id中去除test_id构建train_idx
        train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)
        # 获取训练集
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        # 获取测试集，将正常点与异常点上下拼接构造测试集(正常点在上异常点在下)
        test_x = np.concatenate((normal_data[test_idx], anom_data))
        test_y = np.concatenate((normal_label[test_idx], anom_label))
        
    elif dataset == 'SATELLITE':
        data = loadmat('data/SATELLITE/satellite.mat')
        label = data['y'].astype('float32').squeeze()
        data = data['X'].astype('float32')
        normal_data = data[label == 0]
        normal_label = label[label == 0]
        anom_data = data[label == 1]
        anom_label = label[label == 1]
        train_idx = np.random.choice(np.arange(0, len(normal_data)), 4000, replace=False)
        test_idx = np.setdiff1d(np.arange(0, len(normal_data)), train_idx)
        train_x = normal_data[train_idx]
        train_y = normal_label[train_idx]
        test_x = normal_data[test_idx]
        test_y = normal_label[test_idx]
        test_idx = np.random.choice(np.arange(0, len(anom_data)), int(len(test_x)), replace=False)
        test_x = np.concatenate((test_x, anom_data[test_idx]))
        test_y = np.concatenate((test_y, anom_label[test_idx])) 
                
    train_x, train_y, val_x, val_y, test_x, test_y = split_data(all_train_x=train_x,
                                                                all_train_y=train_y,
                                                                all_test_x=test_x,
                                                                all_test_y=test_y)

    return train_x, train_y, val_x, val_y, test_x, test_y

def pyg_graph_construction(data_name, k, sample_method, proportion, epsilon, seed):

    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(data_name, seed)

    x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, idx = negative_samples(train_x,
                                                                                       train_y,
                                                                                       val_x,
                                                                                       val_y,
                                                                                       test_x,
                                                                                       test_y,
                                                                                       k,
                                                                                       sample_method,
                                                                                       proportion,
                                                                                       epsilon)

    graph = build_graph(x, y, dist, idx, k)

    return graph, train_mask, val_mask, test_mask, test_y


def k_neighbor_pyg_graph_construction(data_name, k_list, sample_method, proportion, epsilon):

    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(data_name)

    k_neighbor_graph_list = []

    for k in k_list:

        x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, idx = negative_samples(train_x,
                                                                                           train_y,
                                                                                           val_x,
                                                                                           val_y,
                                                                                           test_x,
                                                                                           test_y,
                                                                                           k,
                                                                                           sample_method,
                                                                                           proportion,
                                                                                           epsilon)

        graph = build_graph(x, y, dist, idx, k)
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        graph.test_y = test_y
        graph.name = data_name
        k_neighbor_graph_list.append(graph.to(device))

    return k_neighbor_graph_list

if __name__=="__main__":
    pass