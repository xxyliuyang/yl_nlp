from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from graph_classifier import GraphClassifier
from sklearn.metrics import accuracy_score

def create_graph_data():
    dataset = MiniGCDataset(80, 10, 20)
    # 上面参数的意思是生成80个图，每个图的最小节点数>=10, 最大节点数<=20

    # 展示第一条数据
    # graph, label = dataset[10]
    # fig, ax = plt.subplots()
    # nx.draw(graph.to_networkx(), ax=ax)  # 将图转为networkx形式
    # ax.set_title('Class: {:d}'.format(label))
    # plt.show()

    return dataset

def collate(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)

def train():
    # 模型训练
    data_loader = DataLoader(trainset, batch_size=512, shuffle=True, collate_fn=collate)
    model.train()
    epoch_losses = []
    for epoch in range(200):
        epoch_loss = 0
        for iter, (batchg, label) in enumerate(data_loader):
            prediction = model(batchg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

def valid():
    test_loader = DataLoader(testset, batch_size=64, shuffle=False,
                             collate_fn=collate)
    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (batchg, label) in enumerate(test_loader):
            pred = torch.softmax(model(batchg), 1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
    print("accuracy: ", accuracy_score(test_label, test_pred))

if __name__ == '__main__':
    trainset = MiniGCDataset(200, 10, 20)
    testset = MiniGCDataset(100, 10, 20)

    # 构造模型
    model = GraphClassifier(1, 256, trainset.num_classes)
    # 定义分类交叉熵损失
    loss_func = nn.CrossEntropyLoss()
    # 定义Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train()
    valid()