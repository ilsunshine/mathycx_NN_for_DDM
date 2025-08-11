# -*- coding: utf-8 -*-
# import os
# import csv
#
#
# def record_data(file_path, headers, data):
#     """
#     记录实验数据到指定路径的txt或csv文件。
#
#     参数:
#     file_path (str): 文件路径（包括文件名和扩展名），例如："./data/experiment_results.csv"
#     headers (list): 列名列表，例如：["epoch", "loss", "accuracy"]
#     data (list): 数据列表，数据的顺序应该与headers一致，例如：[1, 0.25, 0.98]
#
#     返回:
#     None
#     """
#     # 检查文件是否存在
#     file_exists = os.path.isfile(file_path)
#
#     # 如果文件不存在，创建文件并写入列名
#     if not file_exists:
#         print(f"文件不存在，正在创建新文件：{file_path}")
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         # 判断文件扩展名，处理txt和csv两种情况
#         if file_path.endswith('.csv'):
#             with open(file_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(headers)  # 写入列名
#                 writer.writerow(data)  # 写入第一行数据
#         elif file_path.endswith('.txt'):
#             with open(file_path, mode='w', encoding='utf-8') as f:
#                 f.write("\t".join(headers) + "\n")  # 写入列名，用tab分隔
#                 f.write("\t".join(map(str, data)) + "\n")  # 写入第一行数据，用tab分隔
#         else:
#             print("不支持的文件类型，仅支持 .csv 和 .txt 格式。")
#             return
#     else:
#         # 如果文件存在，直接追加数据
#         print(f"文件已存在，正在追加数据到：{file_path}")
#
#         if file_path.endswith('.csv'):
#             with open(file_path, mode='a', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(data)  # 追加数据行
#         elif file_path.endswith('.txt'):
#             with open(file_path, mode='a', encoding='utf-8') as f:
#                 f.write("\t".join(map(str, data)) + "\n")  # 追加数据行，用tab分隔
#         else:
#             print("不支持的文件类型，仅支持 .csv 和 .txt 格式。")
#             return
#
#     print(f"数据已成功记录到：{file_path}")
#
#
# if __name__=='__main__':
#     # 示例用法
#     headers = ["epoch", "loss", "accuracy"]
#     data = [1, 0.25, 0.98]
#
#     # 调用记录函数，指定文件路径
#     file_path = "./data/experiment_results.csv"
#     record_data(file_path, headers, data)

import os
import csv


# def record_data(file_path, headers, data):
#     """
#     记录实验数据到指定路径的csv文件，支持表头验证与动态更新。
#
#     参数:
#     file_path (str): 文件路径（包括文件名和扩展名），例如："./data/experiment_results.csv"
#     headers (list): 列名列表，例如：["epoch", "loss", "accuracy"]
#     data (list): 数据列表，数据的顺序应该与headers一致，例如：[1, 0.25, 0.98]
#
#     返回:
#     None
#     """
#     # 检查文件是否存在
#     file_exists = os.path.isfile(file_path)
#
#     if not file_exists:
#         # 文件不存在，创建新文件并写入表头和数据
#         print(f"文件不存在，正在创建新文件：{file_path}")
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, mode='w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(headers)  # 写入列名
#             writer.writerow(data)  # 写入第一行数据
#     else:
#         # 文件存在，检查表头并验证一致性
#         print(f"文件已存在，正在验证表头并追加数据到：{file_path}")
#         with open(file_path, mode='r', newline='', encoding='utf-8') as f:
#             reader = csv.reader(f)
#             existing_headers = next(reader)  # 读取现有的表头
#
#         # 找出缺失的列和多余的列
#         missing_headers = [h for h in headers if h not in existing_headers]
#         extra_headers = [h for h in existing_headers if h not in headers]
#
#         if missing_headers or extra_headers:
#             print(f"表头不一致，正在更新文件：缺失列 {missing_headers}, 多余列 {extra_headers}")
#
#             # 更新表头：添加缺失的列，将原表头和新列合并
#             updated_headers = existing_headers + missing_headers
#             updated_data = []
#
#             # 读取现有数据，并为新列填充空值
#             with open(file_path, mode='r', newline='', encoding='utf-8') as f:
#                 reader = csv.DictReader(f)
#                 for row in reader:
#                     # 为每一行补充缺失列
#                     for missing in missing_headers:
#                         row[missing] = ""
#                     updated_data.append(row)
#
#             # 写回文件，更新表头和原数据
#             with open(file_path, mode='w', newline='', encoding='utf-8') as f:
#                 writer = csv.DictWriter(f, fieldnames=updated_headers)
#                 writer.writeheader()  # 写入更新后的表头
#                 writer.writerows(updated_data)  # 写入原数据
#
#         # 追加新的数据行，补充缺失的列
#         with open(file_path, mode='a', newline='', encoding='utf-8') as f:
#             writer = csv.DictWriter(f, fieldnames=updated_headers)
#             row = {header: "" for header in updated_headers}  # 初始化空行
#             for h, d in zip(headers, data):  # 填充现有列的数据
#                 row[h] = d
#             writer.writerow(row)  # 写入新数据行
#
#     print(f"数据已成功记录到：{file_path}")
#
#
# if __name__ == '__main__':
#     # 示例用法
#     headers = ["epoch", "loss", "accuracy"]
#     data = [1, 0.25, 0.98]
#
#     # 调用记录函数，指定文件路径
#     file_path = "./data/experiment_results.csv"
#     record_data(file_path, headers, data)
#
#     # 示例：新的数据列头，含额外和缺失列
#     headers_new = ["epoch", "loss", "lr" ]
#     data_new = [2, 0.22, [0,1]]
#     record_data(file_path, headers_new, data_new)
def record_data(file_path, headers, data):
    """
    记录实验数据到指定路径的csv文件，支持表头验证与动态更新。

    参数:
    file_path (str): 文件路径（包括文件名和扩展名），例如："./data/experiment_results.csv"
    headers (list): 列名列表，例如：["epoch", "loss", "accuracy"]
    data (list): 数据列表，数据的顺序应该与headers一致，例如：[1, 0.25, 0.98]

    返回:
    None
    """
    # 检查文件是否存在
    file_exists = os.path.isfile(file_path)

    if not file_exists:
        # 文件不存在，创建新文件并写入表头和数据
        print(f"文件不存在，正在创建新文件：{file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)  # 写入列名
            writer.writerow(data)  # 写入第一行数据
    else:
        # 文件存在，检查表头并验证一致性
        print(f"文件已存在，正在验证表头并追加数据到：{file_path}")
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            existing_headers = next(reader)  # 读取现有的表头

        # 初始化更新后的表头为现有表头
        updated_headers = existing_headers

        # 找出缺失的列和多余的列
        missing_headers = [h for h in headers if h not in existing_headers]
        extra_headers = [h for h in existing_headers if h not in headers]

        if missing_headers or extra_headers:
            print(f"表头不一致，正在更新文件：缺失列 {missing_headers}, 多余列 {extra_headers}")

            # 更新表头：添加缺失的列，将原表头和新列合并
            updated_headers = existing_headers + missing_headers
            updated_data = []

            # 读取现有数据，并为新列填充空值
            with open(file_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 为每一行补充缺失列
                    for missing in missing_headers:
                        row[missing] = ""
                    updated_data.append(row)

            # 写回文件，更新表头和原数据
            with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=updated_headers)
                writer.writeheader()  # 写入更新后的表头
                writer.writerows(updated_data)  # 写入原数据

        # 如果没有表头不一致的问题，updated_headers 就是现有的表头
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=updated_headers)
            row = {header: "" for header in updated_headers}  # 初始化空行
            for h, d in zip(headers, data):  # 填充现有列的数据
                row[h] = d
            writer.writerow(row)  # 写入新数据行

    print(f"数据已成功记录到：{file_path}")
