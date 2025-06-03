import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
import matplotlib.patches as patches

def visualize_cosine_distance(cosine_distance_table):
    # 创建一个 pandas DataFrame 来存储输入的余弦距离数据
    df = pd.DataFrame(cosine_distance_table)

    # 行列名设置
    df.columns = ['Cpp', 'Java', 'JavaScript', 'Kotlin', 'Python', 'Rust', 'Haskell', 'C', 'Go', 'Swift', 'AppleScript', 'Ruby', 'Raku', 'PHP', 'Fortran', 'Dart', 'Visual Basic', 'Pascal', 'Scala', 'Avg.']
    df.index = ['Cpp', 'Java', 'JavaScript', 'Kotlin', 'Python', 'Rust', 'Haskell', 'C', 'Go', 'Swift', 'AppleScript', 'Ruby', 'Raku', 'PHP', 'Fortran', 'Dart', 'Visual Basic', 'Pascal', 'Scala', 'English']
    
    # 创建注释矩阵（和 df 同尺寸），用于加粗加斜体指定单元格
    annot_matrix = df.copy().astype(str)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iat[i, j]
            if i == 19 and j == 8:
                annot_matrix.iat[i, j] = rf"$\mathbfit{{{value:.2f}}}$"
            elif i == 10 and j == 19:
                annot_matrix.iat[i, j] = rf"$\mathbfit{{{value:.2f}}}$"
            elif i == 16 and j == 19:
                annot_matrix.iat[i, j] = rf"$\mathbfit{{{value:.2f}}}$"
            else:
                annot_matrix.iat[i, j] = f"{value:.2f}"
    
    # 设置 Seaborn 样式
    sns.set(style="white")

    # 创建热图，使用 'coolwarm' 调色板来表示距离
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(df.astype(float), annot=annot_matrix.values, cmap='RdBu_r', vmin=0, vmax=1,
                     norm=PowerNorm(gamma=0.25),
                     cbar_kws={'label': 'Cosine Similarity'}, fmt='', linewidths=0.5)
    
    # 为指定区域加边框
    rect = patches.Rectangle(
        (0 + 0.05, 19 + 0.05), # 列起点、行起点
        19 - 0.1, # 宽度：列数
        1 - 0.1, # 高度：行数
        linewidth = 2.5,
        edgecolor = 'black',
        facecolor = 'none'
    )
    plt.gca().add_patch(rect)

    # 为指定区域加边框
    rect = patches.Rectangle(
        (19 + 0.05, 0 + 0.05), # 列起点、行起点
        1 - 0.1, # 宽度：列数
        19 - 0.1, # 高度：行数
        linewidth = 2,
        edgecolor = 'black',
        facecolor = 'none'
    )
    plt.gca().add_patch(rect)
    
    # 获取热图的 Colorbar（图例）
    cbar = ax.collections[0].colorbar  # 获取 Colorbar

    # 修改图例标签（"Cosine Distance"）的字号
    cbar.ax.yaxis.label.set_size(15)

    # 修改图例刻度（数值）的字号
    cbar.ax.tick_params(labelsize=12)

    # 设置坐标轴标签字号
    ax.set_xticklabels(df.index, fontsize=15, rotation=45, ha='right')  # X轴标签
    ax.set_yticklabels(df.columns, fontsize=15, rotation=0)  # Y轴标签
    ax.tick_params(axis='both', labelsize=15)  # 统一调整刻度字体大小
    
    # 对 "Avg." 进行加粗
    for label in ax.get_xticklabels():
        if label.get_text() == "Avg.":
            label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        if label.get_text() == "Avg.":
            label.set_fontweight('bold')

    # 保存到本地
    plt.savefig("./matrix_RdBu_r.png", dpi=450)

    # 显示热图
    plt.show()

similarity_update = {
  "Cpp": [1.0, 0.56, 0.46, 0.35, 0.26, 0.38, 0.2, 0.42, 0.44, 0.45, 0.24, 0.28, 0.28, 0.33, 0.36, 0.32, 0.18, 0.32, 0.24, 0.337],
  "Java": [0.56, 1.0, 0.43, 0.59, 0.26, 0.31, 0.14, 0.37, 0.44, 0.6, 0.32, 0.36, 0.32, 0.41, 0.26, 0.41, 0.25, 0.25, 0.49, 0.376],
  "JavaScript": [0.46, 0.43, 1.0, 0.41, 0.3, 0.41, 0.28, 0.37, 0.44, 0.26, 0.52, 0.42, 0.38, 0.44, 0.22, 0.46, 0.35, 0.22, 0.26, 0.368],
  "Kotlin": [0.35, 0.59, 0.41, 1.0, 0.26, 0.43, 0.24, 0.17, 0.43, 0.45, 0.28, 0.46, 0.39, 0.5, 0.24, 0.43, 0.23, 0.23, 0.47, 0.364],
  "Python": [0.26, 0.26, 0.3, 0.26, 1.0, 0.24, 0.08, 0.16, 0.44, 0.22, 0.57, 0.56, 0.48, 0.59, 0.28, 0.24, 0.22, 0.21, 0.33, 0.317],
  "Rust": [0.38, 0.31, 0.41, 0.43, 0.24, 1.0, 0.26, 0.32, 0.44, 0.44, 0.22, 0.34, 0.3, 0.28, 0.24, 0.26, 0.12, 0.2, 0.29, 0.304],
  "Haskell": [0.2, 0.14, 0.28, 0.24, 0.08, 0.26, 1.0, 0.08, 0.32, 0.15, 0.16, 0.16, 0.14, 0.2, 0.12, 0.12, 0.08, 0.12, 0.3, 0.175],
  "C": [0.42, 0.37, 0.37, 0.17, 0.16, 0.32, 0.08, 1.0, 0.4, 0.4, 0.28, 0.32, 0.28, 0.36, 0.37, 0.29, 0.2, 0.38, 0.18, 0.297],
  "Go":[0.44, 0.44, 0.44, 0.43, 0.44, 0.44, 0.32, 0.4, 1.0, 0.46, 0.3, 0.42, 0.39, 0.48, 0.32, 0.44, 0.25, 0.31, 0.29, 0.389],
  "Swift": [0.45, 0.6, 0.26, 0.45, 0.22, 0.44, 0.15, 0.4, 0.46, 1.0, 0.3, 0.4, 0.32, 0.46, 0.15, 0.45, 0.26, 0.24, 0.42, 0.357],
  "AppleScript": [0.24, 0.32, 0.52, 0.28, 0.57, 0.22, 0.16, 0.28, 0.3, 0.3, 1.0, 0.24, 0.18, 0.24, 0.16, 0.16, 0.4, 0.15, 0.18, 0.272],
  "Ruby": [0.28, 0.36, 0.42, 0.46, 0.56, 0.34, 0.16, 0.32, 0.42, 0.4, 0.24, 1.0, 0.48, 0.47, 0.16, 0.22, 0.18, 0.16, 0.28, 0.328],
  "Raku": [0.28, 0.32, 0.38, 0.39, 0.48, 0.3, 0.14, 0.28, 0.39, 0.32, 0.18, 0.48, 1.0, 0.38, 0.16, 0.2, 0.2, 0.18, 0.25, 0.295],
  "PHP": [0.33, 0.41, 0.44, 0.5, 0.59, 0.28, 0.2, 0.36, 0.48, 0.46, 0.24, 0.47, 0.38, 1.0, 0.18, 0.3, 0.25, 0.22, 0.31, 0.356],
  "Fortran": [0.36, 0.26, 0.22, 0.24, 0.28, 0.24, 0.12, 0.37, 0.32, 0.15, 0.16, 0.16, 0.16, 0.18, 1.0, 0.2, 0.16, 0.37, 0.24, 0.233],
  "Dart": [0.32, 0.41, 0.46, 0.43, 0.24, 0.26, 0.12, 0.29, 0.44, 0.45, 0.16, 0.22, 0.2, 0.3, 0.2, 1.0, 0.14, 0.26, 0.39, 0.294],
  "Visual Basic": [0.18, 0.25, 0.35, 0.23, 0.22, 0.12, 0.08, 0.2, 0.25, 0.26, 0.4, 0.18, 0.2, 0.25, 0.16, 0.14, 1.0, 0.35, 0.2, 0.223],
  "Pascal": [0.32, 0.25, 0.22, 0.23, 0.21, 0.2, 0.12, 0.38, 0.31, 0.24, 0.15, 0.16, 0.18, 0.22, 0.37, 0.26, 0.35, 1.0, 0.21, 0.243],
  "Scala": [0.24, 0.49, 0.26, 0.47, 0.33, 0.29, 0.3, 0.18, 0.29, 0.42, 0.18, 0.28, 0.25, 0.31, 0.24, 0.39, 0.2, 0.21, 1.0, 0.296],
  "English": [0.04500000000000004, 0.06499999999999995, 0.09499999999999997, 0.10499999999999998, 0.12, 0.07999999999999996, 0.030000000000000027, 0.040000000000000036, 0.07999999999999996, 0.10999999999999999, 0.14500000000000002, 0.13, 0.09999999999999998, 0.08999999999999997, 0.025000000000000022, 0.11499999999999999, 0.15000000000000002, 0.06000000000000005, 0.10499999999999998, 1.0]
}

# 调用函数进行可视化
visualize_cosine_distance(similarity_update)
