# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 13:03:30 2025

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def calcMeanDF(data, index):
    return data.pivot_table(index=index, aggfunc='mean').reset_index()
def custom_boxplot(ax, data, positions, color, label):
    # Set transparency
    alpha = 1
    
    # Draw boxplot, do not show outliers
    box = ax.boxplot(data, positions=positions, 
                    flierprops=dict(markeredgecolor="red"),
                    boxprops=dict(color=color),  # Box color remains unchanged, opaque
                    whiskerprops=dict(color=color, alpha=alpha),  # Whisker transparency 50%
                    capprops=dict(color=color, alpha=alpha),  # Cap transparency 50%
                    medianprops=dict(color="black"),
                    showfliers=False)

    # Get boxplot statistical data
    for i, d in enumerate(data):
        quartiles = np.percentile(d, [25, 50, 75])
        Q1, Q2, Q3 = quartiles[0], quartiles[1], quartiles[2]
        IQR = Q3 - Q1  # Interquartile range

        # Use 3 times IQR to calculate upper and lower bounds for outliers
        lower_bound = Q1 - 12 * IQR
        upper_bound = Q3 + 12 * IQR

        # Mark outliers
        outliers = [x for x in d if x < lower_bound or x > upper_bound]
        for outlier in outliers:
            ax.plot(i + 1, outlier, marker='o', color='red', markersize=4, 
                   markeredgecolor='red', fillstyle='none')  # Use red circles to mark outliers
    
    # Create an opaque line for legend (maintain box color)
    ax.plot([], [], color=color, label=label) 
    return box

    # 绘制箱线图，不显示异常值
    box = ax.boxplot(data, positions=positions, flierprops=dict(markeredgecolor="red"), showfliers=False)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)  # 设置网格线样式
    
    # 获取箱线图的统计数据
    for i, d in enumerate(data):
        quartiles = np.percentile(d, [25, 50, 75])
        Q1, Q2, Q3 = quartiles[0], quartiles[1], quartiles[2]
        IQR = Q3 - Q1  # 四分位距

        # 使用3倍IQR计算异常值的上下限
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # 标记异常值
        outliers = [x for x in d if x < lower_bound or x > upper_bound]
        for outlier in outliers:
            ax.plot(positions[i], outlier, marker='o', color='red', markersize=5, 
                    markeredgecolor='red', fillstyle='none')  # 使用红色圆圈标记异常值

    return box


    alpha = 1
    
    
    box = ax.boxplot(data, positions=positions, 
                    flierprops=dict(markeredgecolor="red"),
                    boxprops=dict(color=color), 
                    whiskerprops=dict(color=color, alpha=alpha),  
                    capprops=dict(color=color, alpha=alpha),  
                    medianprops=dict(color=color),
                    showfliers=False)

    for i, d in enumerate(data):
        quartiles = np.percentile(d, [25, 50, 75])
        Q1, Q2, Q3 = quartiles[0], quartiles[1], quartiles[2]
        IQR = Q3 - Q1  


        lower_bound = Q1 - 12 * IQR
        upper_bound = Q3 + 12 * IQR

        outliers = [x for x in d if x < lower_bound or x > upper_bound]
        for outlier in outliers:
            ax.plot(i + 1, outlier, marker='o', color='red', markersize=4, 
                   markeredgecolor='red', fillstyle='none')  
    
    ax.plot([], [], color=color, label=label) 
    return box

if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 4), constrained_layout=True)
    
    # Load data for the first plot
    n = 500
    data = pd.read_csv("..\\compared_experiment_results_with_trans\\experiment_gaussian-2025-09-17-152358_with_trans_sigma1_1.csv")
    
    data['s1'] = data['s1'].round(2)
    data["n"] = data["n"].round(1)
    sigma_min, sigma_max = 0, 7
    
    
    data2 = pd.read_csv("..\\compared_experiment_results_without_trans\\experiment_gaussian-2025-09-17-170038_without_trans_sigma1_1.csv")
    data2['s1'] = data['s1'].round(2)
    data2["n"] = data["n"].round(1)
    
    # First subfigure
    ax1 = axs[0]
    
    subdata = data.loc[data["sqrt_q"] == 1]
    err="gerr"
    # First data
    
    dd2 = pd.concat([data2["rep_no"], data2["s2"], data2["eigen_mat_" + err]], axis=1)
    d2 = pd.pivot(dd2, index="rep_no", columns="s2", values="eigen_mat_" + err)
    pdd2 = d2.to_dict(orient='list')
    adata2 = [pdd2[key] for key in pdd2.keys()]
    positions2 = np.arange(1, len(adata2) + 1)
    custom_boxplot(ax1, adata2, positions2 - 0.3, color="blue", label='Two-stage approach')
     
    
    # Create pivot table
    dd = pd.concat([subdata["rep_no"], subdata["s2"], subdata["eigen_mat_gerr"]], axis=1)
    d = pd.pivot(dd, index="rep_no", columns="s2", values="eigen_mat_gerr")
    pdd = d.to_dict(orient='list')
    adata = [pdd[key] for key in pdd.keys()]
    # Generate position information with the same dimension as data
    positions = np.arange(1, len(adata) + 1)

    # Draw boxplot
    #custom_boxplot1(ax1, adata, positions)
    custom_boxplot(ax1, adata, positions - 0.3, color="magenta", label='ASE')
    
    
    # Set x-axis labels
    keys_list = list(pdd.keys())
    rounded_keys = [round(float(key), 2) for key in keys_list]
    
    # Correction: correctly set x-axis ticks and labels
    step=5
    ax1.set_xticks(positions[::step])
    ax1.set_xticklabels(rounded_keys[::step], rotation=0)
    ax1.set_title(r'$\sigma_1$=1.0')
    
    # Set titles and labels
    ax1.set_xlabel("$\sigma_2$")
    ax1.set_ylabel(r'$\max_{i\in [n]}\min_{Q\in SE(d)}\Vert \hat{G}_i -QG_i^* \Vert_F$', 
               rotation=90, ha='center', va='center', fontsize=10, labelpad=10)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.3, 1.0), ncol=1)
    
    # Second and third subplots: use original comparison code
    # Load data for the second and third plots
    data1 = pd.read_csv("..\\compared_experiment_results_without_trans\\experiment_gaussian-2025-09-16-004333_without_trans_sigma2_1.csv")
    data1['s1'] = data1['s1'].round(2)
    data1['q'] = data1['q'].round(2)
    data1["sqrt_q"] = data1["sqrt_q"].round(2)
    data1["n"] = data1["n"].round(1)
    data1["eigen_mat_rerr"] = data1["eigen_mat_rerr"].round(2)
    data1["eigen_mat_terr"] = data1["eigen_mat_terr"].round(2)

    data4 = pd.read_csv("..\\compared_experiment_results_with_trans\\experiment_gaussian-2025-09-15-231033_with_trans_sigma2_1.csv")
    data4['s1'] = data4['s1'].round(2)
    data4['q'] = data4['q'].round(2)
    data4["sqrt_q"] = data4["sqrt_q"].round(2)
    data4["n"] = data4["n"].round(1)
    data4["eigen_mat_rerr"] = data4["eigen_mat_rerr"].round(2)
    data4["eigen_mat_terr"] = data4["eigen_mat_terr"].round(2)
    
    sigma_min, sigma_max = 0, 26
    data1 = data1[data1["s1"].between(sigma_min, sigma_max)]
    data4 = data4[data4["s1"].between(sigma_min, sigma_max)]
    
    # Second and third subplots
    err="gerr"
    ax = axs[1]  # axs[1] and axs[2]
        
    ax.set_ylim(0, 8)
    
        
    # Process first dataset
    dd1 = pd.concat([data1["rep_no"], data1["s1"], data1["eigen_mat_" + err]], axis=1)
    d1 = pd.pivot(dd1, index="rep_no", columns="s1", values="eigen_mat_" + err)
    pdd1 = d1.to_dict(orient='list')
    adata1 = [pdd1[key] for key in pdd1.keys()]
    positions1 = np.arange(1, len(adata1) + 1)
    custom_boxplot(ax, adata1, positions1 - 0.3, color="blue", label='Two-stage approach')
        
    # Process fourth dataset
    dd4 = pd.concat([data4["rep_no"], data4["s1"], data4["eigen_mat_" + err]], axis=1)
    d4 = pd.pivot(dd4, index="rep_no", columns="s1", values="eigen_mat_" + err)
    pdd4 = d4.to_dict(orient='list')
    adata4 = [pdd4[key] for key in pdd4.keys()]
    positions4 = np.arange(1, len(adata4) + 1)
    custom_boxplot(ax, adata4, positions4 + 0.3, color="magenta", label='ASE')
        
    # Set x-axis labels
    keys_list = list(pdd4.keys())
    rounded_keys = [round(float(key)) for key in keys_list]
        
    # Correction: correctly set x-axis ticks and labels
    step = 6
    ax.set_title(r'$\sigma_2$=1.0')
    ax.set_xticks(positions4[::step])
    ax.set_xticklabels(rounded_keys[::step])
    ax.set_xlabel("$\sigma_1$")    
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.7, 0.18), ncol=1)
        
    
    # Add overall title (optional)
    
    plt.savefig("..\\experiment_figures\\figure_1.pdf")
    plt.show(block=False)