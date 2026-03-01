from clip_demo import search_by_image, load_text_candidates, search_by_text
import os
from glob import glob

# 1. 测试图片->文本检索
print("="*50)
print("测试1：图片检索文本")
print("="*50)

print("正在加载文本候选集...")
text_candidates = load_text_candidates("candidates.txt")
print(f"共加载 {len(text_candidates)} 条文本候选。")
test_image = "images/247248892.jpg"  

if os.path.exists(test_image):
    print(f"正在测试图片: {test_image}")
    results = search_by_image(test_image, text_candidates)
    print(f"\n输入图片: {test_image}")
    print("匹配结果：")
    for i, res in enumerate(results):
        print(f"  {i+1}. {res['text']} (相似度: {res['similarity']:.4f})")

# 2. 测试文本->图片检索
print("\n" + "="*50)
print("测试2：文本检索图片")
print("="*50)

image_paths = glob("images/*.jpg") + glob("images/*.png") + glob("images/*.jpeg")
query = "一只在草地上奔跑的狗"

results = search_by_text(query, image_paths)
print(f"\n输入文本: {query}")
print("匹配结果：")
for i, res in enumerate(results[:3]):  # 只显示前3张
    print(f"  {i+1}. {os.path.basename(res['image_path'])} (相似度: {res['similarity']:.4f})")