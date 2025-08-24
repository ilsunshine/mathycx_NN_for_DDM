import h5py
import os
import re
import numpy as np
import glob
import math
import argparse
import shutil
import random
#import tqdm
from tqdm import tqdm  # 正确导入方式
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from itertools import combinations


def find_duplicate_vectors(h5_path, dataset_name="A", tol=1e-5):
    """
    在 HDF5 文件中查找名为 dataset_name 的重复向量。

    参数：
        h5_path (str): HDF5 文件路径
        dataset_name (str): 数据名称，默认为 "A"
        tol (float): 重复阈值，相对距离小于此值视为重复

    输出：
        打印报告，若有重复则输出重复对及其路径和误差
    """

    def collect_datasets(h5obj, path=''):
        """递归收集所有名为 dataset_name 的数据集和路径"""
        datasets = []
        for key in h5obj:
            item = h5obj[key]
            full_path = f"{path}/{key}"
            if isinstance(item, h5py.Group):
                datasets.extend(collect_datasets(item, full_path))
            elif isinstance(item, h5py.Dataset) and key == dataset_name:
                data = item[()]
                if data.ndim == 1:  # 确保是向量
                    datasets.append((full_path, data))
        return datasets

    # 打开文件并收集所有相关数据集
    with h5py.File(h5_path, 'r') as f:
        all_datasets = collect_datasets(f)

    print(f"共找到 {len(all_datasets)} 个名为 '{dataset_name}' 的向量数据。\n")

    # 分组后两两比较
    has_duplicate = False
    # class_path=[]#根据重复性进行简单聚类
    # class_path_data=[]
    # class_number=[]#记录每个类重复了多少次
    for (path1, vec1), (path2, vec2) in combinations(all_datasets, 2):
        if vec1.shape != vec2.shape:
            continue
        norm = np.linalg.norm(vec1)
        if norm == 0:
            continue  # 避免除以零
        rel_error = np.linalg.norm(vec1 - vec2) / norm
        if rel_error < tol:
            has_duplicate = True
            print(" 检测到重复向量：")
            print(f" - 路径1: {path1}")
            print(f" - 路径2: {path2}")
            print(f" - 相对误差: {rel_error:.2e}\n")
    #     #进行聚类
    #     new_class = True
    #     for i in range(len(class_path_data)):
    #         if vec1.shape != class_path_data[i].shape:
    #             continue
    #         rel_error = np.linalg.norm(class_path_data[i] - vec2) / norm
    #         if rel_error < tol:
    #             new_class = False
    #             class_number[i]+=1
    #             continue #该类已经被记录过
    #     if new_class:
    #         class_path.append(path1)
    #         class_path_data.append(vec1)
    #         class_number.append(1)
    # print(f"经过简单聚类得到该项数据可以分为{len(class_path_data)}类,距离中心的路劲为{class_path},每类的重复次数分别为{class_number}")





    if not has_duplicate:
        print(" 未检测到重复向量。")


# import h5py
# import numpy as np


def cluster_duplicate_vectors(h5_path, dataset_name="A", tol=1e-5, verbose=True):
    """
    对 HDF5 文件中所有名为 dataset_name 的向量数据进行重复性检测并聚类。

    参数：
        h5_path (str): HDF5 文件路径
        dataset_name (str): 数据名称，默认为 "A"
        tol (float): 相对距离阈值，小于该值视为重复
        verbose (bool): 是否打印详细报告

    返回：
        clusters (list of dict): 每个聚类的结构化信息列表
    """

    def collect_datasets(h5obj, path=''):
        """递归收集所有名为 dataset_name 的一维向量及其路径"""
        datasets = []
        for key in h5obj:
            item = h5obj[key]
            full_path = f"{path}/{key}"
            if isinstance(item, h5py.Group):
                datasets.extend(collect_datasets(item, full_path))
            elif isinstance(item, h5py.Dataset) and key == dataset_name:
                data = item[()]
                if data.ndim == 1:
                    datasets.append((full_path, data))
        return datasets

    with h5py.File(h5_path, 'r') as f:
        all_datasets = collect_datasets(f)

    if verbose:
        print(f"Total {len(all_datasets)} vectors named '{dataset_name}' found.\n")

    clusters = []  # 每个聚类为一个 dict
    assigned = set()  # 已被归类的路径

    for path1, vec1 in all_datasets:
        if path1 in assigned:
            continue

        current_cluster = {
            "center_path": path1,
            "center_vector": vec1,
            "vector_length": len(vec1),
            "members": [path1],
            "norms":np.linalg.norm(vec1)
        }
        assigned.add(path1)

        for path2, vec2 in all_datasets:
            if path2 in assigned or path1 == path2 or vec1.shape != vec2.shape:
                continue
            norm = np.linalg.norm(vec1)
            if norm == 0:
                continue
            rel_error = np.linalg.norm(vec1 - vec2) / norm
            if rel_error < tol:
                current_cluster["members"].append(path2)
                assigned.add(path2)

        clusters.append(current_cluster)

    # 打印报告
    if verbose:
        print(f"Clustering completed. {len(clusters)} clusters found.\n")
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i + 1}:")
            print(f"  Center path: {cluster['center_path']}")
            print(f"  Vector length: {cluster['vector_length']}")
            print(f"  Number of members: {len(cluster['members'])}")
            print(f" norm of Center:{cluster['norms']}")
            if len(cluster["members"]) > 1:
                print("  Member paths:")
                for p in cluster["members"]:
                    print(f"    - {p}")
            print()

    return clusters

def pad_to(A, H_max, W_max, alpha):#将矩阵A进行边界零延拓到（H_max+2*alpha)*(W_max+2*alph)的维度
    h, w = A.shape
    dh = H_max - h
    dw = W_max - w
    pad_top1 = dh // 2
    pad_bottom1 = dh - pad_top1
    pad_left1 = dw // 2
    pad_right1 = dw - pad_left1

    pad_top = pad_top1 + alpha
    pad_bottom = pad_bottom1 + alpha
    pad_left = pad_left1 + alpha
    pad_right = pad_right1 + alpha

    return np.pad(A, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)


def process_hdf5_subdomains(file_path, alpha=0, save=False):#weight_function进行重新排列与延拓，使得每个data.h5文件的weight_function的维度一致
    A_list = []
    meta_list = []  # 保存坐标范围等元数据
    with h5py.File(file_path, 'r+' if save else 'r') as f:
        subdomain_group = f['sub_domain']
        keys = sorted(subdomain_group.keys(), key=lambda x: int(x.split('_')[-1]))  # e.g., subdomain_0, subdomain_1...
        # print(keys)
        # 第一次遍历：构建所有矩阵 & 记录最大尺寸
        for key in keys:
            g = subdomain_group[key]
            coords = g['coordinates'][()]     # shape (N_i, 2)
            weights = g['weight_function'][()]  # shape (N_i,)

            x_vals = np.unique(coords[:, 0])
            y_vals = np.unique(coords[:, 1])
            W, H = len(x_vals), len(y_vals)

            x2j = {x: j for j, x in enumerate(x_vals)}
            y2i = {y: i for i, y in enumerate(y_vals)}

            A = np.zeros((H, W), dtype=weights.dtype)
            for (x, y), w in zip(coords, weights):
                i = y2i[y]
                j = x2j[x]
                A[i, j] = w

            A_list.append(A)
            meta_list.append((x_vals, y_vals))

        # 记录最大尺寸
        H_max = max(A.shape[0] for A in A_list)
        W_max = max(A.shape[1] for A in A_list)

        # 第二次遍历：延拓 & 可选写入
        padded_list = []
        for A, (x_vals, y_vals), key in zip(A_list, meta_list, keys):
            A_pad = pad_to(A, H_max, W_max, alpha)
            padded_list.append(A_pad)

            if save:
                g = subdomain_group[key]
                if 'A_i' in g:
                    del g['A_i']
                if 'weight_function_grid' in g:
                    del g['weight_function_grid']
                g.create_dataset('weight_function_grid', data=A_pad)

                # 坐标边界
                x_min, x_max = float(x_vals[0]), float(x_vals[-1])
                y_min, y_max = float(y_vals[0]), float(y_vals[-1])

                for name, val in zip(['x_min', 'x_max', 'y_min', 'y_max'], [x_min, x_max, y_min, y_max]):
                    if name in g:
                        del g[name]
                    g.create_dataset(name, data=val)

    return padded_list  # 返回所有 A_i（已经延拓）
def dataset_fingerprint(dataset):
    """生成数据集的唯一指纹（类型+形状+数据哈希）"""
    try:
        data = dataset[()]
        # 统一处理标量数据为numpy数组
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # 生成数据哈希
        blake = hashlib.blake2b()
        blake.update(data.tobytes())
        data_hash = blake.hexdigest()

        return (str(dataset.dtype), str(dataset.shape), data_hash)

    except Exception as e:
        print(f"Error processing {dataset.name}: {str(e)}")
        return None


def find_duplicate_datasets(hdf_file):
    """查找重复数据集的核心函数"""
    fingerprints = defaultdict(list)

    def _scan_groups(group):
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                # 生成数据集指纹
                fp = dataset_fingerprint(item)
                if fp:
                    fingerprints[fp].append(item.name)
            elif isinstance(item, h5py.Group):
                _scan_groups(item)

    _scan_groups(hdf_file)
    return {k: v for k, v in fingerprints.items() if len(v) > 1}


def analyze_duplicates(duplicates):
    """分析重复数据集并生成报告"""
    report = []
    for (dtype, shape, _), paths in duplicates.items():
        entry = {
            "count": len(paths),
            "paths": paths,
            "dtype": dtype,
            "shape": shape,
            "is_vector": eval(shape) == (len(eval(shape)),)  # 判断是否为向量
        }
        report.append(entry)

    # 按重复数量排序
    return sorted(report, key=lambda x: (-x['count'], x['dtype']))
def data_repeatability(file_path):#检验hdf5文件中是否有重复的数据
    try:
        with h5py.File(file_path, 'r') as hdf_file:
            duplicates = find_duplicate_datasets(hdf_file)

            if not duplicates:
                print("\n未发现重复数据集")
                exit()

            analysis = analyze_duplicates(duplicates)
            print("\n重复数据集分析报告：")
            print("=" * 60)

            for entry in analysis:
                vector_info = "[向量]" if entry['is_vector'] else ""
                print(f"重复数量: {entry['count']} | 数据类型: {entry['dtype']} | "
                      f"形状: {entry['shape']} {vector_info}")
                print("重复路径:")
                print("\n".join(f"  • {p}" for p in entry['paths']))
                print("-" * 60)

    except Exception as e:
        print(f"错误: {str(e)}")
def print_hdf5_structure(filepath):
    """兼容Windows控制台的HDF5结构打印函数"""

    def _print_group(group, indent):
        for name, obj in group.items():
            if isinstance(obj, h5py.Group):
                print(f"{' ' * indent}+-- {name}/")  # 改用ASCII符号
                _print_group(obj, indent + 4)
            else:
                shape = obj.shape if obj.shape else "(scalar)"
                dtype = obj.dtype
                print(f"{' ' * indent}+-- {name} [Shape: {shape}, Dtype: {dtype}]")

    with h5py.File(filepath, 'r') as f:
        print(f"File: {filepath}")  # 移除Unicode符号
        print("+-- /")  # 根节点标识
        _print_group(f, indent=4)


def print_scalar_datasets(hdf5_path):
    """
    打印hdf5文件下所有的标量值
    :param hdf5_path:
    :return:
    """
    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.shape == ():
            print(f"{name}: {obj[()]}")  # 读取标量值

    with h5py.File(hdf5_path, "r") as f:
        f.visititems(visit_func)

def copy_group(src_group, dest_group):
    """递归复制源Group中的所有数据集、子组和属性到目标Group"""
    # 复制属性
    for attr_name in src_group.attrs:
        dest_group.attrs[attr_name] = src_group.attrs[attr_name]

    # 复制所有子元素
    for name in src_group:
        obj = src_group[name]
        if isinstance(obj, h5py.Group):
            # 创建子组并递归复制
            new_group = dest_group.create_group(name)
            copy_group(obj, new_group)
        else:
            # 复制数据集及其属性
            data = obj[()]
            dset = dest_group.create_dataset(name, data=data)
            for attr_name in obj.attrs:
                dset.attrs[attr_name] = obj.attrs[attr_name]

def is_valid_hdf5(filepath):
    """验证文件是否为有效的HDF5文件"""
    try:
        with h5py.File(filepath, 'r'):
            return True
    except:
        return False


def process_subdirectories(input_path, output_dir):
    """处理所有data+编号子目录"""
    os.makedirs(output_dir, exist_ok=True)

    # 初始化错误日志
    error_log = os.path.join(output_dir, "processing_errors.txt")
    with open(error_log, "w") as f:
        f.write("故障文件记录：\n")

    for entry in os.listdir(input_path):
        if re.fullmatch(r'data\d+', entry):
            print(f"正在处理{r'data+', entry}")
            subdir_path = os.path.join(input_path, entry)
            if os.path.isdir(subdir_path):
                data_number = re.search(r'\d+', entry).group()
                output_path = os.path.join(output_dir, f'data{data_number}.h5')

                try:
                    with h5py.File(output_path, 'w') as f_out:
                        # 处理GlobalData.h5
                        global_file = os.path.join(subdir_path, 'GlobalData.h5')
                        if os.path.exists(global_file):
                            if is_valid_hdf5(global_file):  # 新增校验
                                with h5py.File(global_file, 'r') as f_global:
                                    if 'GlobalData' in f_global:
                                        global_group = f_out.create_group('GlobalData')
                                        copy_group(f_global['GlobalData'], global_group)
                            else:
                                raise RuntimeError(f"损坏的GlobalData文件: {global_file}")

                        # 处理subdomain文件
                        subdomain_files = [
                            f for f in os.listdir(subdir_path)
                            if re.match(r'subdomain_\d+_0\.h5', f)
                        ]

                        if not subdomain_files:
                            with open(error_log, "a") as log:
                                log.write(f"[缺失subdomain] {subdir_path}\n")
                            continue

                        subdomain_group = f_out.create_group('sub_domain')
                        for file in subdomain_files:
                            sub_filepath = os.path.join(subdir_path, file)
                            if os.path.exists(sub_filepath) and is_valid_hdf5(sub_filepath):  # 新增校验
                                try:
                                    with h5py.File(sub_filepath, 'r') as f_sub:
                                        if 'subdomain_data' in f_sub:
                                            sub_num = re.search(r'\d+', file).group()
                                            sub_group = subdomain_group.create_group(f'subdomain_{sub_num}')
                                            copy_group(f_sub['subdomain_data'], sub_group)
                                except Exception as e:
                                    with open(error_log, "a") as log:
                                        log.write(f"[处理失败] {sub_filepath} - {str(e)}\n")
                            else:
                                with open(error_log, "a") as log:
                                    log.write(f"[无效文件] {sub_filepath}\n")

                except Exception as main_error:
                    with open(error_log, "a") as log:
                        log.write(f"[严重错误] 处理目录 {subdir_path} 失败: {str(main_error)}\n")
                    continue  # 关键错误后继续处理其他目录
# 使用示例



def isolate_problem_files(log_path, output_dir, problem_dir):
    """
    将错误日志中记录的问题文件移动到隔离目录
    :param log_path: 错误日志文件路径
    :param output_dir: 原始输出目录
    :param problem_dir: 问题文件存放目录
    """
    # 创建问题文件存放目录
    os.makedirs(problem_dir, exist_ok=True)
    #os.makedirs(log_path,exist_ok=True)
    # 从日志中提取所有问题data编号
    pattern = r'data(\d+)'  # 匹配data后跟数字的编号
    problem_numbers = set()  # 使用集合避免重复

    # with open(log_path, 'r', encoding='utf-8') as f:
    with open(log_path, "r", encoding="gbk") as f:
        for line in f:
            # 匹配两种可能的日志格式
            matches = re.findall(r'data(\d+)(?=[\\/]|$)', line)
            if matches:
                problem_numbers.update(matches)

    # 移动问题文件
    moved_files = []
    for num in problem_numbers:
        src_file = os.path.join(output_dir, f'data{num}.h5')
        if os.path.exists(src_file):
            dest_file = os.path.join(problem_dir, f'data{num}.h5')
            try:
                shutil.move(src_file, dest_file)
                moved_files.append(src_file)
                print(f"已移动: {src_file} -> {dest_file}")
            except Exception as e:
                print(f"移动失败 {src_file}: {str(e)}")
        else:
            print(f"文件不存在: {src_file}")

    # 生成报告
    report = os.path.join(problem_dir, "migration_report.txt")
    with open(report, 'w') as f:
        f.write("已隔离的问题文件：\n")
        f.write("\n".join(moved_files))

    print(f"\n操作完成，共移动 {len(moved_files)} 个文件")
    print(f"详细记录见: {report}")
#



def extend_hdf5_datasets(data_dir, alpha, fill_value=0):
    """
    处理所有data*.h5文件，统一扩展weight_function的维度

    :param data_dir: 数据文件存储路径
    :param alpha: 扩展系数（0.1表示增加10%长度）
    :param fill_value: 填充的默认值
    """
    # 获取所有目标文件
    file_pattern = f"{data_dir}/data*.h5"
    files = glob.glob(file_pattern)

    for file_path in files:
        with h5py.File(file_path, "r+") as f:
            # 获取所有subdomain群组
            subdomains = list(f["sub_domain"].keys())
            if not subdomains:
                print(f"⚠️ 跳过空文件: {file_path}")
                continue

            # 计算当前文件的最大维度
            max_length = max(
                f[f"sub_domain/{sub}/weight_function"].shape[0]
                for sub in subdomains
            )
            target_length = math.ceil(max_length * (1 + alpha))

            # 遍历所有subdomain进行扩展
            for sub in subdomains:
                group = f[f"sub_domain/{sub}"]
                dset = group["weight_function"]
                original_data = dset[:]

                # 计算需要填充的长度
                pad_length = target_length - original_data.shape[0]
                if pad_length <= 0:
                    continue

                # 执行填充操作
                padded_data = np.pad(
                    original_data,
                    (0, pad_length),
                    mode="constant",
                    constant_values=fill_value
                )

                # 删除旧数据集并创建新数据集
                del group["weight_function"]
                new_dset = group.create_dataset(
                    "weight_function",
                    data=padded_data
                )

                # 保留原始属性
                for key in dset.attrs:
                    new_dset.attrs[key] = dset.attrs[key]

            print(f"成功处理: {file_path} (新维度: {target_length})")


def split_h5_files(
        source_dir: str,
        train_dir: str,
        val_dir: str,
        split_ratio: float = 0.8,
        copy_mode: bool = True,
        random_seed: int = None
) -> tuple:
    """
    将HDF5文件按比例分配到训练集和验证集目录

    :param source_dir: 源目录路径（包含.h5文件）
    :param train_dir: 训练集目标目录
    :param val_dir: 验证集目标目录
    :param split_ratio: 训练集比例（默认0.8）
    :param copy_mode: True复制文件，False移动文件（默认True）
    :param random_seed: 随机种子（默认None）
    :return: 元组（训练集数量, 验证集数量）

    :raises ValueError: 参数不合法时抛出
    :raises FileNotFoundError: 路径不存在时抛出
    """
    # 参数有效性校验
    if not (0 <= split_ratio <= 1):
        raise ValueError("分割比例必须在0到1之间")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"源目录不存在: {source_dir}")

    # 获取所有.h5文件（忽略子目录）
    h5_files = glob.glob(os.path.join(source_dir, "*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"源目录中没有.h5文件: {source_dir}")

    # 设置随机种子保证可复现
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(h5_files)

    # 计算分割点（保证至少1个文件在验证集）
    split_idx = max(1, int(len(h5_files) * split_ratio))
    train_files = h5_files[:split_idx]
    val_files = h5_files[split_idx:]

    # 创建目标目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 执行文件操作
    op_func = shutil.copy2 if copy_mode else shutil.move
    for file_list, dest_dir in [(train_files, train_dir), (val_files, val_dir)]:
        for src_path in file_list:
            try:
                op_func(src_path, dest_dir)
            except Exception as e:
                print(f"操作失败 {src_path} -> {dest_dir}: {str(e)}")

    return (len(train_files), len(val_files))
def triang_plot(coords,values):#画散点图工具
    """

    :param coords: 二维坐标[N,2]
    :param values: 函数值[N]
    :return:
    """
    x = coords[:, 0]
    y = coords[:, 1]

    # 散点图
    triang = tri.Triangulation(x, y)

    plt.figure(figsize=(8, 6))
    plt.tricontourf(triang, values, cmap='viridis')  # 填充等高线
    plt.colorbar(label='Weight Function Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Triangulated Scalar Field')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
def move_files_by_ratio(source_dir, target_dir, ratio=0.2, seed=42, file_ext=None):
    """
    将源目录中的文件按比例随机移动到目标目录

    参数：
    - source_dir: 源目录路径
    - target_dir: 目标目录路径
    - ratio: 移动比例 (0.0-1.0)
    - seed: 随机种子 (保证可重复性)
    - file_ext: 指定文件扩展名 (如 '.jpg')，None表示所有文件
    """
    # 参数校验
    assert 0 <= ratio <= 1, "移动比例必须在0到1之间"
    assert os.path.isdir(source_dir), f"源目录不存在: {source_dir}"

    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)

    # 获取文件列表（可选过滤扩展名）
    all_files = [f for f in os.listdir(source_dir)
                 if os.path.isfile(os.path.join(source_dir, f))]

    if file_ext:
        all_files = [f for f in all_files if f.lower().endswith(file_ext.lower())]

    if not all_files:
        print(f"警告: 源目录中没有找到符合条件的文件 ({file_ext or '任意类型'})")
        return

    # 设置随机种子保证可重复性
    random.seed(seed)

    # 计算需要移动的文件数量
    move_count = int(len(all_files) * ratio)
    if move_count < 1:
        move_count = 1  # 至少移动1个文件

    # 随机选择文件
    selected_files = random.sample(all_files, move_count)

    # 移动文件（带进度条）
    moved_files = []
    for filename in tqdm(selected_files, desc='移动文件中'):
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)

        try:
            # 处理文件名冲突
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dst_path):
                    new_name = f"{base}_{counter}{ext}"
                    dst_path = os.path.join(target_dir, new_name)
                    counter += 1

            shutil.move(src_path, dst_path)
            moved_files.append((src_path, dst_path))
        except Exception as e:
            print(f"\n移动失败: {filename} - {str(e)}")

    # 结果报告
    print(f"\n完成: 共移动 {len(moved_files)}/{len(all_files)} 个文件")
    print(f"源目录剩余文件: {len(all_files) - len(moved_files)}")
    print(f"目标目录文件总数: {len(os.listdir(target_dir))}")


def rename_and_move_files(input_dir, output_dir, number):
    """
    将input_dir中的datax.h5文件移动到output_dir并重命名为data(x+number).h5
    用于合并数据集，避免重复命名

    参数:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        number (int): 要添加到文件名中的数字
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.startswith('data') and filename.endswith('.h5'):
            try:
                # 提取x的数字部分
                x_str = filename[4:-3]  # 去掉'data'和'.h5'
                x = int(x_str)

                # 计算新文件名
                new_x = x + number
                new_filename = f'data{new_x}.h5'

                # 构建完整路径
                src_path = os.path.join(input_dir, filename)
                dst_path = os.path.join(output_dir, new_filename)

                # 移动并重命名文件
                shutil.copy2(src_path, dst_path)
                print(f"Moved and renamed: {filename} -> {new_filename}")

            except ValueError:
                print(f"Skipping {filename}: could not extract number")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


def move_back_h5_files(input_dir, output_dir, number, number_inf, number_sup):
    """
    将input_dir中数字编号在[number_inf, number_sup]范围内的datax.h5文件
    移动到output_dir，并重命名为data(x-number).h5

    参数:
        input_dir (str): 当前存放文件的目录
        output_dir (str): 要移动回去的目标目录
        number (int): 要减去的数字量
        number_inf (int): 编号下限(包含)
        number_sup (int): 编号上限(包含)
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.startswith('data') and filename.endswith('.h5'):
            try:
                # 提取文件名的数字部分
                base = filename[4:-3]  # 去掉"data"和".h5"
                x = int(base)

                # 检查数字是否在指定范围内
                if number_inf <= x <= number_sup:
                    # 计算新文件名
                    new_x = x - number
                    new_filename = f'data{new_x}.h5'

                    # 构造完整路径
                    src_path = os.path.join(input_dir, filename)
                    dst_path = os.path.join(output_dir, new_filename)

                    # 移动文件
                    shutil.move(src_path, dst_path)
                    print(f'已移动并重命名: {filename} -> {new_filename}')

            except ValueError:
                print(f'跳过文件 {filename} - 无法解析数字部分')
            except Exception as e:
                print(f'处理文件 {filename} 时出错: {str(e)}')


def get_h5_file_paths(folder_path, prefix='data', suffix='.h5'):#获取目录下所有前缀为prefix的hd5文件
    pattern = os.path.join(folder_path, f"{prefix}*.h5")
    file_list = sorted(glob.glob(pattern))
    return file_list
# 使用示例
# if __name__ == "__main__":
#     isolate_problem_files(
#         log_path="D:/pycharm_project/NN_for_DDM/output/processing_errors.log",
#         output_dir="D:/pycharm_project/NN_for_DDM/output",
#         problem_dir="D:/pycharm_project/NN_for_DDM/problem_files"
#     )
def add_data_to_hdf5(data_dir):#增加部分数据结构测试效果
    """
        处理所有data*.h5文件，增加部分数据结构

        :param data_dir: 数据文件存储路径
        """
    # 获取所有目标文件
    file_pattern = f"{data_dir}/data*.h5"
    files = glob.glob(file_pattern)
    print("向GlobalData中增加数据kappah, kappaH,sigmah,sigmaH")
    print("向数据集里增加数据为 子域的中心点坐标center_x, center_y,宽高x_length,y_length ")
    for file_path in files:
        with h5py.File(file_path, "r+") as f:
            # 获取所有subdomain群组
            subdomains = list(f["sub_domain"].keys())
            if not subdomains:
                print(f" !!!️ 跳过空文件: {file_path}")
                continue

            # 计算当前文件的最大维度
            max_length = max(
                f[f"sub_domain/{sub}/weight_function"].shape[0]
                for sub in subdomains
            )
            global_group=f[f"GlobalData"]
            H=global_group["H"][()]
            h=global_group['h'][()]
            kappa=global_group["kappa"][()]
            kappah=kappa*h
            kappaH=kappa*H
            sigma=global_group["sigma"][()]
            sigmah=sigma*h
            sigmaH=sigma*H

            for name, val in zip(['kappah', 'kappaH',"sigmah","sigmaH" ],
                                 [kappah,kappaH,sigmah,sigmaH]):
                if name in global_group:
                    del global_group[name]
                global_group.create_dataset(name, data=val)

            # 遍历所有subdomain进行扩展
            for sub in subdomains:
                group = f[f"sub_domain/{sub}"]
                x_min=group["x_min"][()]
                x_max=group["x_max"][()]
                y_min=group["y_min"][()]
                y_max=group['y_max'][()]
                center_x=(x_min+x_max)/2
                center_y=(y_min+y_max)/2
                x_length=x_max-x_min
                y_length=y_max-y_min
                # x_min, x_max = float(x_vals[0]), float(x_vals[-1])
                # y_min, y_max = float(y_vals[0]), float(y_vals[-1])

                for name, val in zip(['center_x', 'center_y', 'x_length', 'y_length'], [center_x, center_y, x_length, y_length]):
                    if name in group:
                        del group[name]
                    group.create_dataset(name, data=val)
                # dset = group["weight_function"]
                # original_data = dset[:]
                #
                # # 计算需要填充的长度
                # pad_length = target_length - original_data.shape[0]
                # if pad_length <= 0:
                #     continue

                # 执行填充操作
                # padded_data = np.pad(
                #     original_data,
                #     (0, pad_length),
                #     mode="constant",
                #     constant_values=fill_value
                # )
                #
                # # 删除旧数据集并创建新数据集
                # del group["weight_function"]
                # new_dset = group.create_dataset(
                #     "weight_function",
                #     data=padded_data
                # )

            #     # 保留原始属性
            #     for key in dset.attrs:
            #         new_dset.attrs[key] = dset.attrs[key]
            #
            # #print(f"成功处理: {file_path} (新维度: {target_length})")
if __name__ == "__main__":
    #数据整合
    # input_path = r'D:\pycharm_project\NN_for_DDM\data\low_frequency_k2\low_frenquency_k2'  # 替换为你的输入路径
    # output_dir = r'D:\pycharm_project\NN_for_DDM\data\low_frequency_k2_interatation'  # 替换为你的输出目录
    # process_subdirectories(input_path=input_path, output_dir=output_dir)
    # split_ratio = 0.9#训练数据占所有数据的比例
    # #
    # # # #隔离错误文件
    # isolate_problem_files(
    #     log_path=os.path.join(output_dir,"processing_errors.txt"),
    #     output_dir=output_dir,
    #     problem_dir=os.path.join(output_dir,"problem_data")
    # )
    #
    # # #process_hdf5_subdomains(file_path=output_dir)
    # # #打印数据结构
    # # # file_path = r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\train_data\data6.h5"  # 替换为你的HDF5文件路径
    # # # print_hdf5_structure(file_path)
    # # # #
    # # # 处理weight_function延拓为weight_function_grid
    # path_dir=output_dir
    # file_list=get_h5_file_paths(path_dir)
    # for file_dir in file_list:
    #     print(file_dir)
    #     # print_hdf5_structure(file_dir)
    #     A=process_hdf5_subdomains(file_path=file_dir,alpha=2,save=True)
    # #增加部分参数
    # add_data_to_hdf5(output_dir)
    # # #分配数据进行训练
    # train_num, val_num = split_h5_files(
    #     source_dir=output_dir,
    #     train_dir=os.path.join(output_dir,'train_data'),
    #     val_dir=os.path.join(output_dir,'test_data'),
    #     split_ratio=split_ratio,
    #     copy_mode=True,
    #     random_seed=42#设置随机数种子保证可复现性
    # )
    #
    # print(f"分配完成：训练集 {train_num} 个，验证集 {val_num} 个")



    #合并数据集
    # rename_and_move_files(input_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\train_data",
    #                       output_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_k_plus_interation\train_data",
    #                       number=1000)
    # rename_and_move_files(input_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_k2_interatation\train_data",
    #                       output_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_k_plus_interation\train_data",
    #                       number=0)
    #
    # rename_and_move_files(input_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\test_data",
    #                       output_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_k_plus_interation\test_data",
    #                       number=1000)
    # rename_and_move_files(input_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_k2_interatation\test_data",
    #                       output_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_k_plus_interation\test_data",
    #                       number=0)



    # process_subdirectories(input_path=input_path,output_dir=output_dir)
    #检验参数结构

    # #重复数据检验
    file_path = r"D:\pycharm_project\NN_for_DDM\data\low_frequency_k2_interatation\data0.h5" # 替换为你的HDF5文件路径
    data_repeatability(file_path)
    #
    # #分配数据进行训练
    # train_num, val_num = split_h5_files(
    #     source_dir="./data/low_frequency_interatation",
    #     train_dir="./data/low_frequency_interatation/train_data",
    #     val_dir="./data/low_frequency_interatation/test_data",
    #     split_ratio=0.8,
    #     copy_mode=True,
    #     random_seed=42
    # )
    #
    # print(f"分配完成：训练集 {train_num} 个，验证集 {val_num} 个")

    #对所有的文件进行将weight_function转化为矩阵并做延拓
    # path_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation"
    # file_list=get_h5_file_paths(path_dir)
    # for file_dir in file_list:
    #     print(file_dir)
    #     print_hdf5_structure(file_dir)
        # A=process_hdf5_subdomains(file_path=file_dir,alpha=2,save=True)

    #减少数据
    #move_files_by_ratio(source_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\train_data",target_dir=r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\备用数据",ratio=0.4)
# 测试数据整合的代码

    #打印数据结构
    file_path = r"D:\pycharm_project\NN_for_DDM\data\low_frequency_k2_interatation\data0.h5"  # 替换为你的HDF5文件路径
    print_hdf5_structure(file_path)
    print_scalar_datasets(file_path)


    #add_data_to_hdf5( r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\备用数据")
    # add_data_to_hdf5(r"D:\pycharm_project\NN_for_DDM\data\low_frequency_interatation\test_data")
    # print_hdf5_structure(file_path)
    # print_scalar_datasets(file_path)