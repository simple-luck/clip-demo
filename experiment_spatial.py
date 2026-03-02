import os
import matplotlib
matplotlib.use('Agg')  # 无图形界面时仍可保存图片
import matplotlib.pyplot as plt
import numpy as np
from clip_demo import search_by_image

# ==================== 配置 ====================
# 测试图片路径（请根据实际情况修改）
image_paths = {
    'cat_left_dog_right': 'images/微信图片_20260302110242_252_1.jpg',
    'bird_above_cat': 'images/微信图片_20260302110416_253_1.jpg',
    'person_infront_car': 'images/微信图片_20260302110631_254_1.jpg',
    # 可以继续添加更多图片
}

# 每个图片对应的对比文本
tests = [
    {
        'img_key': 'cat_left_dog_right',
        'texts': {
            'correct': 'a cat on the left and a dog on the right',
            'wrong': 'a dog on the left and a cat on the right',
            'no_spatial': 'a cat and a dog'
        }
    },
    {
        'img_key': 'bird_above_cat',
        'texts': {
            'correct': 'a bird above a cat',
            'wrong': 'a cat above a bird',
            'no_spatial': 'a bird and a cat'
        }
    },
    {
        'img_key': 'person_infront_car',
        'texts': {
            'correct': 'a person in front of a car',
            'wrong': 'a car in front of a person',
            'no_spatial': 'a person and a car'
        }
    }
]

# 创建保存结果的目录
os.makedirs('results/exp_spatial', exist_ok=True)

# ==================== 运行实验 ====================
print("=" * 60)
print("实验一：空间关系理解测试")
print("=" * 60)

summary = []  # 存储汇总数据

for test in tests:
    img_key = test['img_key']
    img_path = image_paths.get(img_key)
    if not img_path or not os.path.exists(img_path):
        print(f"警告：图片 {img_path} 不存在，跳过")
        continue

    # 准备文本
    text_dict = test['texts']
    text_list = list(text_dict.values())
    text_names = list(text_dict.keys())

    # 调用检索函数，获取所有文本的相似度
    matches = search_by_image(img_path, text_list, top_k=len(text_list))
    sim_dict = {m['text']: m['similarity'] for m in matches}

    # 输出结果
    print(f"\n正在测试图片：{img_key}")
    row = {'image': img_key}
    for name in text_names:
        text = text_dict[name]
        sim = sim_dict[text]
        print(f"  {name:12s} : {text[:30]}... 相似度 = {sim:.4f}")
        row[name] = sim
    summary.append(row)

    # 判断正确文本是否排名第一
    best_text = matches[0]['text']
    if best_text == text_dict['correct']:
        print("  ✅ 正确描述排名第一")
    else:
        print("  ❌ 正确描述未排名第一")

    # ========== 可视化 ==========
    plt.figure(figsize=(6, 4))
    x = np.arange(len(text_names))
    colors = ['green' if name == 'correct' else 'orange' for name in text_names]
    plt.bar(x, [row[name] for name in text_names], color=colors)
    plt.xticks(x, text_names, rotation=15)
    plt.ylabel('Cosine Similarity')
    plt.title(f'Spatial Relations: {img_key}')
    plt.tight_layout()
    plt.savefig(f'results/exp_spatial/{img_key}.png', dpi=150)
    plt.close()
    print(f"  图表已保存至 results/exp_spatial/{img_key}.png")

# ==================== 汇总打印 ====================
print("\n" + "=" * 60)
print("实验一汇总：")
for row in summary:
    print(f"{row['image']}: 正确={row['correct']:.4f}, 错误={row['wrong']:.4f}, 无空间={row['no_spatial']:.4f}")