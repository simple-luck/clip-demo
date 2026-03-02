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
从该数据集中随机挑选出20张图片放在images文件夹下作为测试图片，
从数据集的评论中随机抽取30条不重复的作为文本候选集，加上自己对某个图片的一个精准评价（以验证模型确实能将该文本与图片匹配）。

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
### 📊 实验结果
### 以文搜图
输入文本：“A man holding a fish”

输出结果：
<img width="1553" height="414" alt="image" src="https://github.com/user-attachments/assets/ae72bfbe-833e-4e45-8703-8008156e54fd" />

### 以图搜文
输入图片：<img width="915" height="693" alt="image" src="https://github.com/user-attachments/assets/ca5844dc-afb2-4a69-91f4-5e2cf3660293" />

输出结果：

  1. A man with a white hat and sunglasses holding a fish. (相似度: 0.3234)
     
  2. Two young men are having good time. (相似度: 0.1977)
     
  3. A man with glasses holds a small dark-haired child. (相似度: 0.1880)
     
  4. A boy is kayaking on a river. (相似度: 0.1870)
     
  5. A person in a cowboy hat rides a tan horse. (相似度: 0.1723)


## 模型性能探索
为了进一步了解clip模型的性能，我在使用CLIP进行图文检索时，简单设计了三组对比实验：空间关系测试、细粒度类别区分、物体部件识别测试。结果发现，CLIP对‘左边/右边’的区分能力弱，对细粒度品种差异较敏感，对物体部件识别能力一般。
### 空间关系测试
<img width="1732" height="1161" alt="image" src="https://github.com/user-attachments/assets/6af9fb32-a384-42a5-b81a-3333b2fa7add" />

### 细粒度类别区分
<img width="1659" height="1040" alt="image" src="https://github.com/user-attachments/assets/25d24b72-729e-42a3-bfe0-77427e488540" />

### 物体部件识别
<img width="1723" height="743" alt="image" src="https://github.com/user-attachments/assets/48226d84-3a63-431f-bc15-a3a10136fb44" />


### 综合分析
CLIP在细粒度类别区分上表现尚可，但在空间关系理解和部件识别上存在明显不足。这说明：

CLIP的视觉编码器（ViT/ResNet）擅长捕捉全局语义和类别特征，但对局部空间结构和细粒度细节的建模能力有限。

其训练目标为图文全局匹配，导致模型倾向于学习“图像整体”与“文本整体”的关联，而忽略了图像内部的局部关系。
 
## 🔍 对多模态学习的思考
通过该项目，我对多模态学习有了更深的理解：
1. 多模态学习是指利用来自不同模态，如文本、图像、音频、视频、传感器数据等）信息，通过模型学习它们之间的关联与互补性，完成联合推理或生成任务。常见的应用有图文检索、视觉问答、跨模态生成、情感分析、医疗诊断等场景。
   
2.CLIP的核心原理：CLIP的核心是一个双塔结构，由一个图像编码器（如ViT或ResNet）和一个文本编码器（如Transformer）构成，分别将图像和文本映射到同一特征空间。采用对比训练的方式，通过4亿图文对训练，拉近匹配的图文对，推远不匹配的对，这种对比损失使得模型能够理解图文之间的语义关联。

3.零样本学习的能力：CLIP的强大之处在于它将视觉概念与语言描述直接关联，无需针对特定任务微调，即可实现零样本分类和检索。

4. 未来探索方向：我关注到clip模型在空间关系理解和部件识别上存在明显不足，而袁粒老师的VOLO论文正是针对这一问题：VOLO通过Outlook注意力机制显式建模局部细粒度特征，能够更好地捕捉图像中的边缘、纹理、部件等局部结构。因此或许将VOLO引入CLIP的视觉编码器，能增强模型对空间关系的感知，提升对物体部件的识别能力，在不损失全局语义的前提下，补充局部细节信息，从而改善图文匹配的准确性。。

## 📚 参考资料
1. CLIP原始论文  
2. HuggingFace CLIP文档
3. 袁粒老师代表作：VOLO, T2T-ViT
