import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import glob
import pandas as pd


def collect_csv_files(directory):
    """
    收集指定目录下所有的CSV文件路径

    参数:
    directory -- 要搜索的目录

    返回:
    csv_files -- 包含所有CSV文件路径的列表
    """
    # 确保目录路径以斜杠结尾
    if not directory.endswith('/'):
        directory += '/'

    # 使用glob模块搜索所有CSV文件
    csv_files = glob.glob(directory + '**/*.csv', recursive=True)

    # 排序文件路径以便于阅读
    csv_files.sort()

    return csv_files

def print_node_attributes(xfg, output_path, nodes_df_dict):
    """
    打印XFG图中所有节点的属性，并保存到文件

    参数:
    xfg -- NetworkX DiGraph 对象
    output_path -- 输出文件的路径
    """


    with open(output_path, 'w') as f:
        f.write("XFG Node Attributes:\n")
        f.write(f"Total nodes: {len(xfg.nodes)}\n")
        f.write("=" * 50 + "\n\n")

        for node in xfg.nodes():
            code = nodes_df_dict[node]
            f.write(f"Node: {node}\n")
            f.write(f"Code: {code}\n")
            attributes = xfg.nodes[node]
            if attributes:
                for key, value in attributes.items():
                    # 尝试以可读性更好的格式输出复杂对象
                    try:
                        if isinstance(value, (dict, list)):
                            value_str = json.dumps(value, indent=2)
                        else:
                            value_str = str(value)
                        f.write(f"  {key}: {value_str}\n")
                    except:
                        f.write(f"  {key}: [Complex object that cannot be displayed]\n")
            else:
                f.write("  No attributes\n")
            f.write("\n")

        # 添加边的信息
        f.write("Edge Attributes:\n")
        f.write(f"Total edges: {len(xfg.edges)}\n")
        f.write("=" * 50 + "\n\n")

        for edge in xfg.edges():
            f.write(f"Edge: {edge[0]} -> {edge[1]}\n")
            attributes = xfg.edges[edge]
            if attributes:
                for key, value in attributes.items():
                    try:
                        if isinstance(value, (dict, list)):
                            value_str = json.dumps(value, indent=2)
                        else:
                            value_str = str(value)
                        f.write(f"  {key}: {value_str}\n")
                    except:
                        f.write(f"  {key}: [Complex object that cannot be displayed]\n")
            else:
                f.write("  No attributes\n")
            f.write("\n")


# def visualize_xfg(xfg, output_path, nodes_df_dict):
#     """
#     可视化 XFG 图并保存为图像文件
#
#     参数:
#     xfg -- NetworkX DiGraph 对象
#     output_path -- 输出图像的路径
#     """
#     plt.figure(figsize=(12, 8))
#
#     # 使用 spring_layout 布局算法
#     pos = nx.spring_layout(xfg, seed=42)
#
#     # 绘制节点
#     nx.draw_networkx_nodes(xfg, pos, node_size=500, node_color='lightblue')
#
#     # 绘制边，使用箭头表示方向
#     nx.draw_networkx_edges(xfg, pos, arrowsize=20, width=1.5, edge_color='gray')
#
#     # 绘制节点标签
#     nx.draw_networkx_labels(xfg, pos, font_size=10)
#
#     # 移除坐标轴
#     plt.axis('off')
#
#     # 保存图像
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()  # 关闭图形以释放内存


import matplotlib.pyplot as plt
import networkx as nx

def visualize_xfg(xfg, output_path, nodes_df_dict):
    """
    可视化 XFG 图并保存为图像文件

    参数:
    xfg -- NetworkX DiGraph 对象
    output_path -- 输出图像的路径
    nodes_df_dict -- 节点号到代码的映射字典，例如 {1: "code1", 13: "code13", ...}
    """
    plt.figure(figsize=(12, 8))

    # 使用 spring_layout 布局算法
    pos = nx.spring_layout(xfg, seed=42)

    # 绘制节点
    nx.draw_networkx_nodes(xfg, pos, node_size=500, node_color='lightblue')

    # 绘制边，使用箭头表示方向
    nx.draw_networkx_edges(xfg, pos, arrowsize=20, width=1.5, edge_color='gray')

    # 创建自定义标签：节点号 + 对应的代码
    labels = {node: f"{node}\n{nodes_df_dict[node]}" for node in xfg.nodes()}

    # 绘制节点标签，调整字体大小和位置
    nx.draw_networkx_labels(xfg, pos, labels=labels, font_size=8)

    # 移除坐标轴
    plt.axis('off')

    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

from os.path import join, exists
from typing import List, Set, Tuple, Dict
def read_csv(csv_file_path: str) -> List:
    """
    read csv file
    """
    assert exists(csv_file_path), f"no {csv_file_path}"
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data

def process_xfg_files(input_dir, output_dir, max_files=None):
    """
    处理目录下所有 .pkl 文件，生成可视化图像和节点属性信息

    参数:
    input_dir -- 包含 .pkl 文件的目录
    output_dir -- 输出图像和属性信息的目录
    max_files -- 最大处理文件数，None表示处理所有文件
    """


    directory = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/val/'
    csv_files = collect_csv_files(directory)


    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 递归查找所有 .pkl 文件
    pkl_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))

    # 限制处理文件数
    if max_files is not None and max_files > 0:
        pkl_files = pkl_files[:max_files]

    print(f"找到 {len(pkl_files)} 个 .pkl 文件，将处理 {len(pkl_files)} 个")

    # 处理每个 .pkl 文件
    for pkl_file in tqdm(pkl_files):
            print("\n\n\n**********pkl_file:", pkl_file)
        # try:
            # 读取 XFG 图
            xfg = nx.read_gpickle(pkl_file)

            # 创建输出文件路径
            rel_path = os.path.relpath(pkl_file, input_dir)
            base_name = os.path.splitext(rel_path)[0]
            index_name = rel_path.split('/')[0]

            # if index_name != '169996':
            #     continue
            nodes_csv =""
            for one_csv_name in csv_files:
                if str(index_name) in  one_csv_name:
                    if 'node' in one_csv_name:
                        nodes_csv = one_csv_name
                        break

            print("\nnodes_csv:", nodes_csv)
            nodes_df = read_csv(nodes_csv)
            # nodes_df = nodes_df.reset_index(drop=True)  # drop=True 丢弃旧索引

            nodes_df_dict = {}
            ii = 0
            for  row in nodes_df:
                row_code = row['code']
                nodes_df_dict[ii] = row_code
                ii += 1


            print("\nos.path.dirname(rel_path):", os.path.dirname(rel_path))
            print("\nbase_name:", base_name)
            # 确保输出文件的目录存在
            output_dir_path = os.path.join(output_dir, os.path.dirname(rel_path))
            os.makedirs(output_dir_path, exist_ok=True)

            # 可视化 XFG 图
            viz_output_path = os.path.join(output_dir, f"{base_name}.png")
            visualize_xfg(xfg, viz_output_path, nodes_df_dict)

            # 保存节点属性信息
            attr_output_path = os.path.join(output_dir, f"{base_name}_attributes.txt")
            print_node_attributes(xfg, attr_output_path, nodes_df_dict)

            print(f"已处理: {pkl_file}")
            print(f"  - 可视化: {viz_output_path}")
            print(f"  - 属性信息: {attr_output_path}")

        # except Exception as e:
        #     print(f"处理 {pkl_file} 时出错: {e}")




if __name__ == "__main__":
    # 输入和输出目录
    input_directory = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG"
    output_directory = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_analysis"

    # 处理文件，可以限制处理文件数量以便于测试
    # 比如只处理5个文件
    process_xfg_files(input_directory, output_directory, max_files=500)

    print("处理完成!")