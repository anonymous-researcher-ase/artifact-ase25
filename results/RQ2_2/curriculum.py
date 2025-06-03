import matplotlib.pyplot as plt

# 数据
languages = ["AppleScript", "Python", "Swift", "Kotlin", "JavaScript", "Go", "Rust", "Java", "C++", "Haskell"]
original_0 = [10.68, 21.54, 14.77, 15.35, 18.67, 16.73, 12.91, 18.23, 14.04, 10.53]
original = [14.68, 25.54, 18.77, 19.35, 22.67, 20.73, 16.91, 22.23, 18.04, 14.53]
near_to_far = [24.15, 31.73, 29.21, 29.03, 28.56, 31.51, 28.33, 26.83, 28.19, 22.75]
random = [23.73, 31.23, 28.07, 28.45, 27.27, 29.85, 25.89, 26.20, 27.98, 22.06]
far_to_near = [23.46, 29.57, 27.66, 27.87, 26.54, 29.07, 26.12, 25.63, 27.47, 21.52]

axis_label_fontsize = 24  # 坐标轴标签字号
ticks_fontsize = 18       # 坐标轴刻度字号
legend_fontsize = 13.5      # 图例字号

# 自定义颜色
color_original = '#3a6f97'    # 蓝色
color_random = '#fb8828'      # 黄色
color_far_to_near = '#c83e32' # 红色
color_near_to_far = '#147e54' # 绿色

# 画图
plt.figure(figsize=(10, 8.5))
plt.plot(languages, original, marker='o', label='Original', color=color_original, markerfacecolor=color_original, markeredgecolor='black', markeredgewidth=1.2, linewidth=2.5, markersize=12)
plt.plot(languages, random, marker='^', label='Random', color=color_random, markerfacecolor=color_random, markeredgecolor='black', markeredgewidth=1.2, linewidth=2.5, markersize=12)
plt.plot(languages, far_to_near, marker='*', label='Far-to-Near', color=color_far_to_near, markerfacecolor=color_far_to_near, markeredgecolor='black', markeredgewidth=1.2, linewidth=2.5, markersize=16)
plt.plot(languages, near_to_far, marker='s', label='Near-to-Far', color=color_near_to_far, markerfacecolor=color_near_to_far, markeredgecolor='black', markeredgewidth=1.2, linewidth=2.5, markersize=12)


# 图例和标签
plt.ylabel("CodeBLEU Score", fontsize=axis_label_fontsize)
plt.xticks(rotation=45, fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.ylim(14, 32.5)
plt.legend(fontsize=legend_fontsize, loc="upper right", ncol=2)
plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)

plt.tight_layout()

# 保存图表
plt.savefig("./Curriculum_modify.png", dpi=300)

# 显示图表
plt.show()
