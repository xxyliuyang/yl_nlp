from dgl.nn.pytorch import GraphConv
import dgl
import torch.nn as nn
import torch.nn.functional as F

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GraphClassifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)  # 定义第一层图卷积
        self.conv2 = GraphConv(hidden_dim, hidden_dim)  # 定义第二层图卷积
        self.classify = nn.Linear(hidden_dim, n_classes)  # 定义分类器

    def forward(self, g):
        """
        g表示批处理后的大图，N表示大图的所有节点数量，n表示图的数量
        """
        # 我们用节点的度作为初始节点特征。对于无向图，入度 = 出度
        h = g.in_degrees().view(-1, 1).float()  # [N, 1]
        # 执行图卷积和激活函数
        h = F.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv2(g, h))  # [N, hidden_dim]
        g.ndata['h'] = h  # 将特征赋予到图的节点
        # 通过平均池化每个节点的表示得到图表示
        hg = dgl.mean_nodes(g, 'h')  # [n, hidden_dim]
        return self.classify(hg)  # [n, n_classes]
