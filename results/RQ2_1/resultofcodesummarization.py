import numpy as np
import matplotlib.pyplot as plt

def plot_bar_chart(ax, x, base_values, java_tuned, python_tuned, bar_width, colors, fontsize):
    """绘制柱状图并标注数值"""
    bars1 = ax.bar(x - bar_width, base_values, width=bar_width, label="Base", color=colors["base"], alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x, java_tuned, width=bar_width, label="Java-SFT", color=colors["java_bar"], alpha=0.8, edgecolor="black", linewidth=1.5)
    bars3 = ax.bar(x + bar_width, python_tuned, width=bar_width, label="Python-SFT", color=colors["python_bar"], alpha=0.8, edgecolor="black", linewidth=1.5)

    # 数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.2f}",
                    ha="center", va="bottom", fontsize=fontsize["label"])
    return bars1, bars2, bars3

def plot_line_chart(ax, x, shifted_x, java_ratio, python_ratio, colors, fontsize):
    """绘制比例提升折线图并标注数值"""
    ax.plot(x, java_ratio, marker="o", linestyle="-", color=colors["java_line"], label="Java-SFT IR", linewidth=2.5, zorder=3)
    ax.plot(shifted_x, python_ratio, marker="o", linestyle="-", color=colors["python_line"], label="Python-SFT IR", linewidth=2.5, zorder=3)

    for i in range(len(x)):
        ax.text(x[i], java_ratio[i] + 0.5, f"{java_ratio[i]:.2f}%", ha="center", fontsize=fontsize["label"])
        ax.text(shifted_x[i], python_ratio[i] + 0.5, f"{python_ratio[i]:.2f}%", ha="center", fontsize=fontsize["label"])

def configure_axes(ax1, ax2, x, categories, fontsize):
    """配置坐标轴与图例等属性"""
    ax1.set_ylabel("BLEU Score", fontsize=fontsize["axis_label"])
    ax1.set_ylim(5, 18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=fontsize["axis_label"])
    ax1.tick_params(axis='y', labelsize=fontsize["ticks"])
    # ax1.legend(loc="upper left", fontsize=fontsize["legend"])

    ax2.set_ylabel("Improvement (%)", fontsize=fontsize["axis_label"])
    ax2.set_ylim(0, 40)
    ax2.tick_params(axis='y', labelsize=fontsize["ticks"])
    # ax2.legend(loc="upper right", fontsize=fontsize["legend"])

    # 设置图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax2.legend(all_handles, all_labels, loc="upper right", fontsize=fontsize["legend"])

    ax1.set_zorder(1)
    ax2.set_zorder(2)
    ax1.patch.set_visible(False)

def draw_summarization_chart():
    # 数据定义
    categories = ["Kotlin", "Haskell", "Swift", "AppleScript"]
    base_values = [15.74, 14.91, 12.66, 10.38]
    java_tuned = [17.06, 16.27, 15.17, 10.97]
    python_tuned = [16.51, 15.94, 14.89, 11.24]
    java_ratio = [8.39, 9.12, 19.83, 5.68]
    python_ratio = [4.89, 6.91, 17.61, 8.29]
    x = np.arange(len(categories))
    shifted_x = x + 0.25

    # 样式参数
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

    # 绘图流程
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    plot_bar_chart(ax1, x, base_values, java_tuned, python_tuned, bar_width, colors, fontsize)
    plot_line_chart(ax2, x, shifted_x, java_ratio, python_ratio, colors, fontsize)
    configure_axes(ax1, ax2, x, categories, fontsize)

    # 保存图像与显示
    plt.tight_layout()
    plt.savefig("Resultofcodesummarization.pdf", bbox_inches = "tight")
    plt.show()

# 执行绘图函数
draw_summarization_chart()