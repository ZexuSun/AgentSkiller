import json
import matplotlib.pyplot as plt
import numpy as np

ratios = []

for line in open("outputs/evaluation/action_results_0120.jsonl", "r"):
    try:
        data = json.loads(line)
        ratios.append(data["evaluators"]["action"]["details"]["param_match_stats"]["full_match_rate"])
    except Exception as e:
        print(data)
        continue

# 统计所有不同的full_match_rate的取值以及数量
unique_vals, counts = np.unique(ratios, return_counts=True)

labels = [f"{x:.2f}" for x in unique_vals]
explode = [0.05] * len(labels)  # 稍微分开各个扇区

plt.figure(figsize=(7, 7))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, counterclock=False)
plt.title("Full Match Rate Distribution (Pie Chart)")
plt.tight_layout()
plt.show()