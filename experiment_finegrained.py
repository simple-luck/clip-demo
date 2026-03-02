import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from clip_demo import search_by_image

# ==================== 配置 ====================
image_paths = {
    'pug': 'images/微信图片_20260302113831_261_1.jpg',
    'beagle': 'images/微信图片_20260302113834_265_1.jpg',
    '747': 'images/微信图片_20260302122117_270_1.jpg',
    'a380': 'images/微信图片_20260302122206_274_1.png',
}

tests = [
    {
        'img_key': 'pug',
        'texts': [
            ('correct', 'a pug dog'),
            ('wrong_similar', 'a beagle dog'),
            ('generic', 'a dog')
        ]
    },
    {
        'img_key': 'beagle',
        'texts': [
            ('correct', 'a beagle dog'),
            ('wrong_similar', 'a pug dog'),
            ('generic', 'a dog')
        ]
    },
    {
        'img_key': '747',
        'texts': [
            ('correct', 'a Boeing 747 airplane'),
            ('wrong_similar', 'an Airbus A380 airplane'),
            ('generic', 'an airplane')
        ]
    },
    {
        'img_key': 'a380',
        'texts': [
            ('correct', 'an Airbus A380 airplane'),
            ('wrong_similar', 'a Boeing 747 airplane'),
            ('generic', 'an airplane')
        ]
    }
]

os.makedirs('results/exp_finegrained', exist_ok=True)

# ==================== 运行实验 ====================
print("=" * 60)
print("实验二：细粒度类别区分测试")
print("=" * 60)

summary = []

for test in tests:
    img_key = test['img_key']
    img_path = image_paths.get(img_key)
    if not img_path or not os.path.exists(img_path):
        print(f"警告：图片 {img_path} 不存在，跳过")
        continue

    # 准备文本
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

    # 判断正确文本是否排名第一
    if matches[0]['text'] == test['texts'][0][1]:
        print("  ✅ 正确描述排名第一")
    else:
        print("  ❌ 正确描述未排名第一")

    # 可视化
    plt.figure(figsize=(6, 4))
    x = np.arange(len(text_names))
    colors = ['green' if name == 'correct' else 'orange' for name in text_names]
    plt.bar(x, [row[name] for name in text_names], color=colors)
    plt.xticks(x, text_names, rotation=15)
    plt.ylabel('Cosine Similarity')
    plt.title(f'Fine-grained: {img_key}')
    plt.tight_layout()
    plt.savefig(f'results/exp_finegrained/{img_key}.png', dpi=150)
    plt.close()
    print(f"  图表已保存至 results/exp_finegrained/{img_key}.png")

print("\n" + "=" * 60)
print("实验二汇总：")
for row in summary:
    print(f"{row['image']}: 正确={row['correct']:.4f}, 相似错误={row['wrong_similar']:.4f}, 泛化={row['generic']:.4f}")