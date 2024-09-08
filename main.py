import socket
import subprocess
import requests
from sklearn.ensemble import RandomForestClassifier
from geoip2.database import Reader
import numpy as np
import time

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

    # 初始化随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# 预测域名是否被墙
def predict_blockage(domain, model, use_proxy=False, proxy_url=None):
    features = build_features(domain, use_proxy, proxy_url)
    prediction = model.predict_proba(features)
    return prediction[0][1]  # 返回被墙的概率

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
