import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from clip_demo import search_by_image

# ==================== 配置 ====================
image_paths = {
    'car': 'images/微信图片_20260302113836_268_1.jpg',
    'bird': 'images/3594029059.jpg',
    'person_glasses': 'images/247248892.jpg',
}

tests = [
    {
        'img_key': 'car',
        'texts': [
            ('whole', 'a car'),
            ('part_present', 'a car wheel'),
            ('part_absent', 'a car door')
        ]
    },
    {
        'img_key': 'bird',
        'texts': [
            ('whole', 'a bird'),
            ('part_present', 'bird wings'),
            ('part_absent', 'bird legs')
        ]
    },
    {
        'img_key': 'person_glasses',
        'texts': [
            ('whole', 'a person'),
            ('part_present', 'glasses'),
            ('part_absent', 'a necklace')
        ]
    }
]

os.makedirs('results/exp_part', exist_ok=True)

# ==================== 运行实验 ====================
print("=" * 60)
print("实验三：物体部件识别测试")
print("=" * 60)

summary = []

for test in tests:
    img_key = test['img_key']
    img_path = image_paths.get(img_key)
    if not img_path or not os.path.exists(img_path):
        print(f"警告：图片 {img_path} 不存在，跳过")
        continue

    text_names = [item[0] for item in test['texts']]
    text_list = [item[1] for item in test['texts']]

    matches = search_by_image(img_path, text_list, top_k=len(text_list))
    sim_dict = {m['text']: m['similarity'] for m in matches}

    print(f"\n正在测试图片：{img_key}")
    row = {'image': img_key}
    for name, text in zip(text_names, text_list):
        sim = sim_dict[text]
        print(f"  {name:15s} : {text[:30]}... 相似度 = {sim:.4f}")
        row[name] = sim
    summary.append(row)

    # 判断整体与部件的差异
    if row['whole'] > row['part_present'] + 0.05:
        print("  📌 整体描述远高于部件描述")
    else:
        print("  🔍 部件描述与整体描述差距不大")

    # 可视化
    plt.figure(figsize=(6, 4))
    x = np.arange(len(text_names))
    colors = ['blue' if name == 'whole' else 'green' if name == 'part_present' else 'gray' for name in text_names]
    plt.bar(x, [row[name] for name in text_names], color=colors)
    plt.xticks(x, text_names, rotation=15)
    plt.ylabel('Cosine Similarity')
    plt.title(f'Part Recognition: {img_key}')
    plt.tight_layout()
    plt.savefig(f'results/exp_part/{img_key}.png', dpi=150)
    plt.close()
    print(f"  图表已保存至 results/exp_part/{img_key}.png")

print("\n" + "=" * 60)
print("实验三汇总：")
for row in summary:
    print(f"{row['image']}: 整体={row['whole']:.4f}, 存在部件={row['part_present']:.4f}, 无关部件={row['part_absent']:.4f}")