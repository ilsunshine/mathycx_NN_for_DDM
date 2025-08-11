# -- coding: utf-8 --
import pandas as pd
import numpy as np
import re


def add_mean_std_to_csv(file_path, metric_columns, param_columns, num_trials):
    """
    处理超参数实验结果，添加均值和标准差,改进版：支持指定超参数列

    参数:
    - file_path: CSV文件路径
    - metric_columns: 需要统计的指标列
    - param_columns: 超参数列名列表（排除无关列）
    - num_trials: 每组参数应有的实验次数
    """
    # 读取数据
    df = pd.read_csv(file_path)

    # 验证分组列是否存在
    missing_params = [col for col in param_columns if col not in df.columns]
    if missing_params:
        raise ValueError(f"参数列不存在：{missing_params}")

    # 分组验证（仅使用超参数列）
    # group_sizes = df.groupby(param_columns).size()
    # if not all(group_sizes == num_trials):
    #     invalid_groups = group_sizes[group_sizes != num_trials]
    #     error_msg = "\n".join([f"- {idx}: {cnt}次" for idx, cnt in invalid_groups.items()])
    #     raise ValueError(f"实验次数错误：\n{error_msg}")

    # 计算统计量
    for metric in metric_columns:
        df[f"{metric}_mean"] = df.groupby(param_columns)[metric].transform('mean')
        df[f"{metric}_std"] = df.groupby(param_columns)[metric].transform('std')

    return df
def copy_csv(source_path, dest_path):#复制csv文件路劲
    with open(source_path, 'r', newline='') as src_file:
        content = src_file.read()
    with open(dest_path, 'w', newline='') as dest_file:
        dest_file.write(content)

def escape_latex(s):
    """
    转义 LaTeX 特殊字符
    """
    if not isinstance(s, str):
        return s
    replace_map = {
        '\\': r'\textbackslash{}',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '#': r'\#',
        '%': r'\%',
        '&': r'\&',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '$': r'\$'
    }
    pattern = re.compile('|'.join(re.escape(key) for key in replace_map))
    return pattern.sub(lambda m: replace_map[m.group()], s)

def csv_to_latex_table(
    csv_path,
    data_columns,
    match_column=None,
    match_value=None,
    float_precision=4
):
    """
    将 CSV 中指定列的数据转化为 LaTeX 表格代码。

    参数：
    - csv_path (str): CSV 文件路径。
    - data_columns (list of str): 需要转化为表格的列名。
    - match_column (str): 用于筛选的列名（可选）。
    - match_value (str or float): 用于筛选的值（可选）。
    - float_precision (int): 浮点数显示精度（默认保留4位小数）。

    返回：
    - str: LaTeX 表格代码。
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 筛选匹配值
    if match_column and match_value is not None:
        df = df[df[match_column] == match_value]

    # 仅保留所需列
    df = df[data_columns]

    # 对齐格式：第一列为 l，其余为 c
    alignments = ['l' if i == 0 else 'c' for i in range(len(data_columns))]
    col_format = ''.join(alignments)

    # 处理表头：进行 LaTeX 特殊字符转义
    escaped_headers = [escape_latex(col) for col in data_columns]
    header_line = ' & '.join(escaped_headers) + r' \\'

    # 构造内容行
    body_lines = []
    for _, row in df.iterrows():
        line = []
        for col in data_columns:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.{float_precision}f}"
            line.append(str(val))
        body_lines.append(' & '.join(line) + r' \\')

    # 拼接 LaTeX 表格代码
    latex_code = "\\begin{tabular}{" + col_format + "}\n"
    latex_code += "\\hline\n"
    latex_code += header_line + "\n"
    latex_code += "\\hline\n"
    latex_code += '\n'.join(body_lines) + "\n"
    latex_code += "\\hline\n\\end{tabular}"

    return latex_code

if __name__ == "__main__":
    #copy_csv(r"D:\pycharm_project\NN_for_DDM\logs\测试initial_cnn_arg结构pooling结构带来的影响\最优数据汇总.csv",r"D:\pycharm_project\NN_for_DDM\logs\测试initial_cnn_arg结构pooling结构带来的影响\最优数据汇总副本.csv")
    # data_path=r"D:\pycharm_project\NN_for_DDM\549logs\测试preditor激活函数与global_pool_size(包含mse时)\最优数据汇总.csv"
    # processed_df = add_mean_std_to_csv(
    #     file_path=data_path,
    #     metric_columns=["最优平均召回率", "最优平均准确度","最优训练准确度","在测试集上的最优测试准确度"],
    #     num_trials=5,
    #     #param_columns=["loss.cross_entropy_loss_with_mse.label_smoothing","loss.cross_entropy_loss_with_mse.alpha","loss.cross_entropy_loss_with_mse.if_update_weights"]
    #     param_columns=["preditor_network.MLP.act_fun","weight_network.AdaptiveCNN.global_pool_size"]
    # )
    #
    # # 保存处理后的结果
    # processed_df.to_csv(f"{data_path}(处理后).csv", index=False)
    # print("处理完成！新增列已添加至原始数据中")
    data_path=r"D:\pycharm_project\NN_for_DDM\最优数据汇总.csv"
    data_columns = ["model.feature_out_dim",'最优训练损失', '最优训练准确度', "在测试集上的最优测试准确度",'最优平均召回率', '最优平均准确度', '最优mAP']
    match_column = 'model.model_network'
    match_value = 'Step_function_network'
    latex = csv_to_latex_table(data_path, data_columns, match_column, match_value)
    print(latex)