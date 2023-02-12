import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
from model import Mymodel
import torch.nn as nn
import numpy as np

path = r'./smartmeter'


### 将分类变量转换成数值型变量
def categorical_numerical(df, col_name):
    categorical_list = list(pd.unique(df[col_name]))
    df[col_name] = df[col_name].map(lambda x: categorical_list.index(x))
    return df


###构建自己的数据集
class Mydataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx, :], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.int64)
        return X, y

    def __len__(self):
        return len(self.y)


###模型的训练
def train(model, writer, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('模型在训练集上的损失:{}'.format(train_loss))
    writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=epoch)
    accuracy = train_correct / len(train_loader.dataset)
    writer.add_scalar(tag='train_acc', scalar_value=accuracy, global_step=epoch)
    print('模型在训练集上的预测精度:{}'.format(accuracy))


###模型的测试
def test(model, writer, device, test_loader, loss_fn, epoch, test_acc):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    print('模型在测试集上的损失:{}'.format(test_loss))
    writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=epoch)
    accuracy = test_correct / len(test_loader.dataset)
    writer.add_scalar(tag='test_acc', scalar_value=accuracy, global_step=epoch)
    print('模型在测试集上的预测精度:{}'.format(accuracy))
    if accuracy > test_acc:
        torch.save(model, './model/BEST_MODEL.pth')
    return accuracy


if __name__ == '__main__':
    sample_submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    test_features = pd.read_csv(os.path.join(path, 'test_features.csv'), encoding='gbk')
    train_features = pd.read_csv(os.path.join(path, 'train_features.csv'))

    ###1. 首先需要填充meter_reading的缺失值(因为缺失值较少，所以直接用均值进行填充)
    test_features['meter_reading'].fillna(test_features['meter_reading'].mean(), inplace=True)
    train_features['meter_reading'].fillna(train_features['meter_reading'].mean(), inplace=True)

    ### 2.将primary use转换为数值型变量
    test_features = categorical_numerical(test_features, 'primary_use')
    train_features = categorical_numerical(train_features, 'primary_use')

    ### 3.删除掉timestamp这一列
    test_features.drop('timestamp', axis=1, inplace=True)
    train_features.drop('timestamp', axis=1, inplace=True)

    ### 4.删除掉非数值型变量
    for col_name in test_features:
        if test_features[col_name].dtype not in ['float64', 'int64']:
            test_features.drop(col_name, axis=1, inplace=True)

    for col_name in train_features:
        if train_features[col_name].dtype not in ['float64', 'int64']:
            train_features.drop(col_name, axis=1, inplace=True)

    ### 4.提取出特征和标签
    y = train_features['anomaly'].values
    X = train_features.drop('anomaly', axis=1).values

    ### 5.划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    ### 6.需要对特征进行标准化（消除量纲的影响）
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    train_dataset = Mydataset(X_train, y_train)
    test_dataset = Mydataset(X_test, y_test)

    ### 7.解决样本不均衡的问题
    class_sample_count = torch.tensor([np.sum((y_train == t)) for t in np.sort(np.unique(y_train))])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in y_train])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, sampler=sampler)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=128)

    ###将过程结果写入tensorboard
    writer = SummaryWriter('logs')

    ###根据电脑配置选择GPU或者cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###神经网络的训练和测试
    model = Mymodel()
    model = model.to(device)

    ###随机梯度下降法
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    ###优化学习率，防止模型不能收敛
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    ###用交叉熵函数作为损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    EPOCH = 10
    test_acc = 0
    for epoch in range(EPOCH):
        print('EPOCH**************************{}/{}'.format(epoch + 1, EPOCH))
        train(model=model, writer=writer, device=device, train_loader=train_dataloader, optimizer=optimizer,
              loss_fn=loss_fn, epoch=epoch)
        test_acc = test(model=model, writer=writer, device=device, test_loader=test_dataloader, loss_fn=loss_fn,
                        epoch=epoch, test_acc=test_acc)
        scheduler.step()
    writer.close()

    ###计算auc
    best_model = torch.load('./model/BEST_MODEL.pth')
    X_test = torch.tensor(X_test.reshape(-1, 1, 49), dtype=torch.float32)
    output = best_model(X_test)
    y_pred = np.reshape(output.argmax(dim=2).numpy(), -1)
    y_true = y_test
    auc = roc_auc_score(y_true, y_pred)
    print('模型在测试集上的auc为{}：'.format(auc))
    acc = np.sum(y_true == y_pred) / len(y_true)
    print('模型在测试集上的acc为{}：'.format(acc))

    ## 预测
    X_pred_ = test_features.values
    X_pred_ = sc.transform(X_pred_)
    X_pred_ = torch.tensor(X_pred_.reshape(-1, 1, 49), dtype=torch.float32)
    output_ = best_model(X_pred_)
    y_pred_ = np.reshape(output_.argmax(dim=2).numpy(), -1)
    sample_submission['anomaly'] = y_pred_
    sample_submission.to_csv('./res.csv', index=False)
