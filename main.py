import socket
import subprocess
import requests
from geoip2.database import Reader
import numpy as np
import time
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # 特征索引
        self.threshold = threshold      # 阈值
        self.left = left                # 左子树
        self.right = right              # 右子树
        self.value = value              # 叶节点的分类值

    def is_leaf_node(self):
        return self.value is not None

# 决策树
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 停止条件
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # 随机选择特征子集
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # 找到最佳分割
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # 递归地创建子树
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        # 遍历每个特征及其可能的阈值
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # 计算信息增益
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # 计算加权平均熵
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # 信息增益
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            # 训练数据自举采样
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # 收集所有树的预测
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 对每个样本的预测进行投票
        return np.swapaxes(tree_preds, 0, 1).mode(axis=1)[0]

# 初始化 geoip2 reader
geoip_reader = Reader('GeoLite2-City.mmdb')  # 下载 GeoLite2-City 数据库文件

# Pathping/traceroute function
def trace_route(domain):
    try:
        # 使用 traceroute（Linux/Mac）或 pathping（Windows）
        output = subprocess.check_output(["tracert", domain], universal_newlines=True)
        ips = []
        for line in output.splitlines():
            if "[" in line and "]" in line:
                ip = line.split("[")[1].split("]")[0]
                ips.append(ip)
        return ips
    except subprocess.CalledProcessError:
        return []

# GeoIP 检查
def check_geoip(ip):
    try:
        response = geoip_reader.city(ip)
        return response.country.iso_code  # 返回国家代码
    except Exception as e:
        print(f"GeoIP 错误: {e}")
        return None

# ICMP ping 检测
def ping(domain):
    try:
        output = subprocess.check_output(["ping", "-c", "4", domain], universal_newlines=True)
        if "0% packet loss" in output:
            return 0  # Ping 成功
        else:
            return 1  # 有丢包，可能被墙
    except Exception:
        return 1  # 发生错误，可能被墙

# HTTP 请求检测 (支持 socks5 代理)
def check_http(domain, use_proxy=False, proxy_url=None):
    try:
        proxies = None
        if use_proxy and proxy_url:
            proxies = {
                'http': f'socks5://{proxy_url}',
                'https': f'socks5://{proxy_url}'
            }
        response = requests.get(f"http://{domain}", timeout=5, proxies=proxies)
        if response.status_code == 200:
            return 0  # HTTP 请求成功
        else:
            return 1  # 非 200 状态码，可能被墙
    except Exception:
        return 1  # 请求失败，可能被墙

# Socket 检测
def check_socket(domain, port=80):
    try:
        sock = socket.create_connection((domain, port), timeout=5)
        sock.close()
        return 0  # Socket 成功连接
    except Exception:
        return 1  # 连接失败，可能被墙

# 构建特征数据
def build_features(domain, use_proxy=False, proxy_url=None):
    features = []
    
    # 获取 traceroute 路径
    ips = trace_route(domain)
    
    # 检查地理位置
    for ip in ips:
        country = check_geoip(ip)
        if country == "CN":
            features.append(1)  # 如果中间有中国 IP，加权为 1
        else:
            features.append(0)

    # 进行多种检测
    features.append(ping(domain))
    features.append(check_http(domain, use_proxy, proxy_url))
    features.append(check_socket(domain))
    
    return np.array(features).reshape(1, -1)

# 训练随机森林模型
def train_model():
    # 构造训练数据（根据历史数据，包含域名的检测情况和是否被墙）
    X_train = [
        # 数据集格式: [是否中国IP, ICMP结果, HTTP结果, Socket结果]
        [1, 1, 1, 1],  # 被墙
        [0, 0, 0, 0],  # 未被墙
        [1, 0, 1, 1],  # 被墙
        [0, 0, 0, 1]  # 未被墙
    ]
    y_train = [1, 0, 1, 0]  # 标签：1 表示被墙，0 表示未被墙

    model = RandomForest(n_trees=100, max_depth=10)
    model.fit(np.array(X_train), np.array(y_train))
    
    return model

# 预测域名是否被墙
def predict_blockage(domain, model, use_proxy=False, proxy_url=None):
    features = build_features(domain, use_proxy, proxy_url)
    prediction = model.predict(features)
    return prediction[0]  # 返回预测结果

if __name__ == "__main__":
    model = train_model()
    
    domain = input("请输入要检测的域名：").strip()
    
    use_proxy = input("是否使用 socks5 代理 (y/n)？").strip().lower() == 'y'
    proxy_url = None
    if use_proxy:
        proxy_url = input("请输入 socks5 代理地址 (格式: 用户名:密码@IP:端口) 或直接 IP:端口：").strip()

    start_time = time.time()
    
    blockage_prob = predict_blockage(domain, model, use_proxy, proxy_url)
    
    print(f"{domain} 被墙的概率为: {blockage_prob * 100:.2f}%")
    print(f"检测完成，耗时 {time.time() - start_time:.2f} 秒")
