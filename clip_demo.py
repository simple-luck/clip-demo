import torch
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# 1. 加载预训练模型
print("正在加载CLIP模型...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"模型加载完成，使用设备: {device}")

# 2. 定义图像编码函数
def encode_image(image_path):
    """
    将图像编码为向量
    """
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"处理图片 {image_path} 出错: {e}")
        return None

# 3. 定义文本编码函数
def encode_text(text_list):
    """
    将文本列表编码为向量
    """
    inputs = processor(text=text_list, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    # 归一化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

# 4. 测试函数
if __name__ == "__main__":
    # 测试单张图片
    test_image = "images/your_test_image.jpg"  # 替换为你的测试图片路径
    if os.path.exists(test_image):
        img_vec = encode_image(test_image)
        print(f"图像向量维度: {img_vec.shape}")
    
    # 测试文本
    texts = ["一只狗", "一只猫", "一辆车"]
    text_vecs = encode_text(texts)
    print(f"文本向量维度: {text_vecs.shape}")

def load_text_candidates(file_path="candidates.txt"):
    """
    从文件加载文本候选集
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        candidates = [line.strip() for line in f.readlines() if line.strip()]
    return candidates

def search_by_image(image_path, text_candidates, top_k=5):
    """
    输入图片，返回最匹配的文本
    """
    # 编码图片
    image_vec = encode_image(image_path)
    if image_vec is None:
        return []
    
    # 编码所有候选文本
    text_vecs = encode_text(text_candidates)
    
    # 计算余弦相似度
    similarities = np.dot(text_vecs, image_vec)
    
    # 获取top_k结果
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': text_candidates[idx],
            'similarity': float(similarities[idx])
        })
    
    return results

def search_by_text(query_text, image_paths, top_k=5):
    """
    输入文本，返回最匹配的图片
    """
    # 编码查询文本
    query_vec = encode_text([query_text])[0]
    
    # 编码所有图片
    image_vecs = []
    valid_paths = []
    
    for img_path in image_paths:
        img_vec = encode_image(img_path)
        if img_vec is not None:
            image_vecs.append(img_vec)
            valid_paths.append(img_path)
    
    if not image_vecs:
        return []
    
    image_vecs = np.array(image_vecs)
    
    # 计算相似度
    similarities = np.dot(image_vecs, query_vec)
    
    # 获取top_k结果
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'image_path': valid_paths[idx],
            'similarity': float(similarities[idx])
        })
    
    return results