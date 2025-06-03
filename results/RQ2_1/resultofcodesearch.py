import numpy as np
import matplotlib.pyplot as plt

def plot_bar_chart(ax, x, base_values, java_tuned, python_tuned, bar_width, colors, fontsize):
    """绘制三个对比柱状图并添加数值标签"""
    bars1 = ax.bar(x - bar_width, base_values, width=bar_width, label="Base", color=colors["base"], alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x, java_tuned, width=bar_width, label="Java-SFT", color=colors["java_bar"], alpha=0.8, edgecolor="black", linewidth=1.5)
    bars3 = ax.bar(x + bar_width, python_tuned, width=bar_width, label="Python-SFT", color=colors["python_bar"], alpha=0.8, edgecolor="black", linewidth=1.5)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:  
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}%", 
                    ha="center", va="bottom", fontsize=fontsize["label"])
    return bars1, bars2, bars3

def plot_line_chart(ax, x, shifted_x, java_ratio, python_ratio, colors, fontsize):
    """绘制 Java 和 Python 的提升比例折线图及标签"""
    ax.plot(x, java_ratio, marker="o", linestyle="-", color=colors["java_line"], label="Java-SFT IR", linewidth=2.5, zorder=3)
    ax.plot(shifted_x, python_ratio, marker="o", linestyle="-", color=colors["python_line"], label="Python-SFT IR", linewidth=2.5, zorder=3)

    # 添加数值标签
    for i in range(len(x)):
        ax.text(x[i], java_ratio[i] + 0.5, f"{java_ratio[i]:.2f}%", ha="center", fontsize=fontsize["label"])
        ax.text(shifted_x[i], python_ratio[i] + 0.5, f"{python_ratio[i]:.2f}%", ha="center", fontsize=fontsize["label"])

def configure_axes(ax1, ax2, x, categories, fontsize):
    """配置坐标轴样式、标签和图例"""
    ax1.set_ylabel("Accuracy (%)", fontsize=fontsize["axis_label"])
    ax1.set_ylim(20, 70)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=fontsize["axis_label"])
    ax1.tick_params(axis='y', labelsize=fontsize["ticks"])
    # ax1.legend(loc="upper left", fontsize=fontsize["legend"])

    ax2.set_ylabel("Improvement (%)", fontsize=fontsize["axis_label"])
    ax2.set_ylim(0, 30)
    ax2.tick_params(axis='y', labelsize=fontsize["ticks"])
    # ax2.legend(loc="upper right", fontsize=fontsize["legend"])

    # 设置图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax2.legend(all_handles, all_labels, loc="upper right", fontsize=fontsize["legend"])

    # 层级设置，确保折线图在柱状图前面
    ax1.set_zorder(1)
    ax2.set_zorder(2)
    ax1.patch.set_visible(False)

def draw_experiment_chart():
    # 数据定义
    categories = ["Kotlin", "Haskell", "Swift", "AppleScript"]
    base_values = [59.6, 51.8, 41.4, 43.2]
    java_tuned = [66.4, 57.6, 45.6, 45.4]
    python_tuned = [63.8, 60.2, 44.6, 46.6]
    java_ratio = [11.41, 11.20, 10.14, 5.09]
    python_ratio = [7.05, 16.22, 7.73, 7.87]
    x = np.arange(len(categories))
    shifted_x = x + 0.25

    # 样式配置
    bar_width = 0.25
    colors = {
        "base": "#d9d9d9",
        "java_bar": "#b0c4de",
        "python_bar": "#ffdab9",
        "java_line": "#2b5d8c",
        "python_line": "#cc7a00"
    }
    fontsize = {
        "title": 28,
        "axis_label": 20,
        "ticks": 15,
        "legend": 15,
        "label": 10
    }

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()

    # 调用绘图函数
    plot_bar_chart(ax1, x, base_values, java_tuned, python_tuned, bar_width, colors, fontsize)
    plot_line_chart(ax2, x, shifted_x, java_ratio, python_ratio, colors, fontsize)
    configure_axes(ax1, ax2, x, categories, fontsize)

    plt.tight_layout()
    # 保存图表
    plt.savefig("Resultofcodesearch.pdf", bbox_inches = "tight")
    # 显示图表
    plt.show()

# 执行绘图
draw_experiment_chart()
