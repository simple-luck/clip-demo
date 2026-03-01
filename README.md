# CLIP 多模态图文检索 Demo

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

## 📝 项目简介

本项目基于 OpenAI 的 **CLIP** 模型（Contrastive Language-Image Pre-training），实现了一个轻量级的**多模态图文检索系统**。CLIP 通过对比学习将图像和文本映射到同一个向量空间，使得跨模态相似度计算成为可能[citation:1][citation:8]。

**核心功能**：
- 🖼️ **以图搜文**：输入图片，返回最匹配的文本描述
- 📝 **以文搜图**：输入文本，返回最匹配的图片

本项目是我为学习多模态深度学习而准备的入门实践。通过这个项目，我深入理解了多模态模型的核心思想，并希望能在研究生阶段继续深入这一方向。

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/你的用户名/clip-multimodal-demo.git
cd clip-multimodal-demo

# 安装依赖
pip install torch torchvision transformers pillow matplotlib
```

### 数据准备
采用Hugging Face 镜像下载Flickr30k数据集，这是专为国内服务器优化的方案，利用 Hugging Face 的镜像站加速下载。

```bash
安装 huggingface_hub：
pip install -U "huggingface_hub[cli]"

设置镜像端点（关键一步，解决国内下载慢的问题）：
export HF_ENDPOINT=https://hf-mirror.com

下载数据集到指定目录：
# 下载到 ~/projects/clip-demo/images/（可以改成自己的路径）
hf download nlphuji/flickr30k --repo-type=dataset --local-dir ~/projects/clip-demo/images//flickr30k
```
从该数据集中随机挑选出20张图片放在images文件夹下作为测试图片
从数据集的评论中随机抽取30条不重复的作为文本候选集。

### 运行实例
```bash
python test_run.py
```
## 📁 项目结构
```bash
clip-demo/
├── clip_demo.py          # 核心代码：模型加载、特征提取、检索函数
├── test_run.py           # 测试脚本
├── visualize.py          # 可视化脚本
├── images/               # 图片文件夹
├── candidates.txt        # 文本候选集
├── requirements.txt      # 依赖包列表
└── README.md             # 项目说明
```

## 💡 核心代码解析
1. 加载预训练模型
```bash
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```
2. 图像特征提取
```bash
def encode_image(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    # 归一化
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()
```
3. 文本特征提取
```bash
def encode_text(text_list):
    inputs = processor(text=text_list, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()
```
4. 相似度计算
采用余弦相似度衡量图文匹配程度：
```bash
similarities = np.dot(text_features, image_features)
```

## 📊 实验结果示例
### 以文搜图
输入文本：“A man holding a fish”
输出结果：
<img width="1553" height="414" alt="image" src="https://github.com/user-attachments/assets/ae72bfbe-833e-4e45-8703-8008156e54fd" />
