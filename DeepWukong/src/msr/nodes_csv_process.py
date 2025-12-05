import os.path
from os.path import join, exists
from typing import List, Set, Tuple, Dict
import pandas as pd
import os
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from sympy.physics.mechanics import potential_energy
from tqdm import tqdm
import json
import glob
import pandas as pd
import re
# from  msr_graph_linevul_intersection import update_joern_nodes_location

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

    # 用字典存储文件
    csv_dict = {}
    v_instance_dict = {}
    for file in csv_files:
        filename = os.path.basename(file)  # 获取文件名
        folder_name = os.path.basename(os.path.dirname(file))
        match = re.findall(r'\d+', folder_name)  # 提取数字
        if match:
            key = match[0]  # 将匹配到的数字转换为整数
            if key in csv_dict:
                csv_dict[key].append(file)  # 若键已存在，添加到列表
            else:
                csv_dict[key] = [file]  # 若键不存在，创建新列表

            if match[-1] == '1':
                if key in v_instance_dict:
                    v_instance_dict[key].append(file)  # 若键已存在，添加到列表
                else:
                    v_instance_dict[key] = [file]  # 若键不存在，创建新列表
    # print("csv_dict:", csv_dict)
    # print("v_instance_dict:", v_instance_dict)
    # print("csv_files:", csv_files)

    return csv_files, csv_dict, v_instance_dict


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

def select_real_code_from_nodes_csv(nodes_csv):
    data = read_csv(nodes_csv)

    data_df = pd.DataFrame(data)
    data_df_dir = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/train/chrome/140744/140744_func_before_target_0.c/'
    data_df.to_excel(os.path.join(data_df_dir, 'nodes_df.xlsx'), index=False)

    code_list = []
    for item in data:
        print(item)
        code_list.append(item['code'])



import re

def get_child_location_from_parent(filename, parent_location, parent_code, child_code):
    # location 形如 '2:2:60:114' -> line 2, column 2, char_offset 60, end_offset 114
    start_line, start_col, start_offset, _ = map(int, parent_location.split(":"))

    # 读取整个文件内容
    with open(filename, "r") as f:
        full_text = f.read()

    # 在 parent_code 中找 child_code 的相对位置
    rel_index = parent_code.find(child_code)
    if rel_index == -1:
        raise ValueError("Child code not found in parent code.")

    # 子 code 在原始文件中的实际 offset
    abs_offset = start_offset + rel_index

    # 找出该 offset 对应的行号和列号
    current_offset = 0
    for line_number, line in enumerate(full_text.splitlines(keepends=True), start=1):
        next_offset = current_offset + len(line)
        if current_offset <= abs_offset < next_offset:
            char_index = abs_offset - current_offset
            return {
                "line": line_number,
                "column": char_index,
                "abs_offset": abs_offset
            }
        current_offset = next_offset

    raise ValueError("Offset exceeds file length.")


import re


def clean_joern_code(code):
    # 去掉最外层引号（CSV 语法）
    if code.startswith('"') and code.endswith('"'):
        code = code[1:-1]

    # 替换 "" 为 " （CSV转义）
    code = code.replace('""', '"')

    # 去掉 ( 和 ) 两边空格
    code = code.replace(" ( ", "(").replace(" )", ")")

    # 逗号后面保留空格，前面不能有空格
    code = code.replace(" , ", ", ")

    # 清理操作符周围多余空格
    ops = ['::', '->', '.', '=', ';', '*', '+', '-', '/', '<', '>', '==', '!=', '>=', '<=', '%']
    for op in ops:
        pattern = r'\s*' + re.escape(op) + r'\s*'
        code = re.sub(pattern, op, code)

    # 多个空格压缩为一个
    code = re.sub(r'\s+', ' ', code)

    return code.strip()


def find_index_from_parent_code(source_code, joern_code_clean):
    import re

    # source_code = '''
    # void GLES2DecoderImpl::DoLinkProgram(GLuint program_id) {
    #   TRACE_EVENT0("gpu", "GLES2DecoderImpl::DoLinkProgram");
    #   Program* program = GetProgramInfoNotShader(
    #       program_id, "glLinkProgram");
    #   if (!program) {
    #     return;
    #   }
    #
    #   // Another call to the same function
    #   Program* program2 = GetProgramInfoNotShader(program_id, "glLinkProgram");
    # }
    # '''
    #
    # joern_code_clean = 'Program*program=GetProgramInfoNotShader(program_id,"glLinkProgram");'

    # 去除空白字符
    source_flat = re.sub(r'\s+', '', source_code)
    joern_flat = re.sub(r'\s+', '', joern_code_clean)

    # 构建 flat_index -> original_index 映射
    index_map = {}
    flat_index = 0
    for i, c in enumerate(source_code):
        if not c.isspace():
            index_map[flat_index] = i
            flat_index += 1

    # 查找所有匹配位置
    matches = [m.start() for m in re.finditer(re.escape(joern_flat), source_flat)]
    # print("matches:", matches)

    if matches:
        # print(f"✅ 找到 {len(matches)} 个匹配项：")
        if len(matches) == 1:

            for idx, match_index in enumerate(matches):
                start_in_original = index_map.get(match_index)
                end_in_original = index_map.get(match_index + len(joern_flat) - 1, len(source_code) - 1)

                matched_snippet = source_code[start_in_original:end_in_original + 1]
                # print(f"\n🔹 匹配 {idx + 1}:")
                # print(f"🧭 原始代码位置：{start_in_original} 到 {end_in_original}")
                # print(f"📌 原始代码片段：\n{matched_snippet}")
                joern_code_clean_new = matched_snippet
        else:
            # matches = [matches[0]]
            start_in_original, end_in_original = 0, 0
            joern_code_clean_new = ""

            for idx, match_index in enumerate(matches):
                start_in_original_tmp = index_map.get(match_index)
                end_in_original_tmp = index_map.get(match_index + len(joern_flat) - 1, len(source_code) - 1)

                matched_snippet = source_code[start_in_original_tmp:end_in_original_tmp + 1]
                # print(f"\n🔹 匹配 {idx + 1}:")
                # print(f"🧭 原始代码位置：{start_in_original} 到 {end_in_original}")
                # print(f"📌 原始代码片段：\n{matched_snippet}")
                # joern_code_clean_new = matched_snippet

                if joern_code_clean_new == "":
                    joern_code_clean_new = matched_snippet
                else:
                    joern_code_clean_new = joern_code_clean_new + "&&&&&" + matched_snippet

                if start_in_original == 0:
                    start_in_original = start_in_original_tmp
                else:
                    start_in_original = str(start_in_original) + "&&&&&" +str(start_in_original_tmp)

                if end_in_original == 0:
                    end_in_original = end_in_original_tmp
                else:
                    end_in_original = str(end_in_original) + "&&&&&" + str(end_in_original_tmp)

    else:
        # print("❌ 未匹配")
        start_in_original, end_in_original = 0, 0
        joern_code_clean_new = ""

    return joern_code_clean_new, start_in_original, end_in_original


# 定义偏移提取函数
def extract_by_offset(filename, start_line, start_column, start_offset, end_offset):
    with open(filename, "r") as f:
        full_text = f.read()

    lines = full_text.splitlines(keepends=True)
    # start_index = sum(len(lines[i]) for i in range(start_line - 1)) + start_column
    start_index = start_offset
    return full_text, full_text[start_index:end_offset]

def update_joern_nodes_location_step_1(joern_nodes, c_file_path):
    joern_nodes_updated = []
    for i in range(len(joern_nodes)):
        # print("i:", i)
        one_node = joern_nodes[i]
        # print("one_node---:   ", one_node)
        one_node_location = one_node['location']
        # print("before clean:", one_node['code'])
        one_node_code = clean_joern_code(one_node['code'])
        # print("after clean:", one_node_code)
        # continue
        one_node_type = one_node['type']
        location_updated = one_node['location_updated']
        code_updated = one_node['code_updated']
        if i == 0:
            one_node['location_updated'] = ""
            one_node['code_updated'] = ""
        else:
            if (len(one_node_code.strip()) >= 0 and one_node_type != 'Symbol' and one_node_type.strip()=='Function') \
                    or (len(one_node_code.strip()) == 0) \
                    or ("&&&&" in location_updated):
                if ":" in one_node_location:
                    line, col, loc_3, end = one_node_location.split(":")
                    parent_location = one_node_location
                    parent_code = one_node['code']
                    parent_type = one_node['type']

                    if "&&&&" in location_updated:
                        start_index, end_index = location_updated.split("#")
                        start_list = start_index.split("&&&&&")
                        end_list = end_index.split("&&&&&")

                        after_joern_preprocess_start_index = int(loc_3)
                        after_joern_preprocess_end_index = int(end)

                        distance_list = []
                        most_close_res_index = -1
                        for iii in range(len(start_list)):
                            one_start = int(start_list[iii])
                            one_end = int(end_list[iii])
                            distance = after_joern_preprocess_start_index - one_start
                            distance_list.append(distance)
                            if distance <= 0:
                                most_close_res_index  = iii -1

                            node_c_code_original_start_index = start_list[most_close_res_index]
                            node_c_code_original_end_index = end_list[most_close_res_index]

                            parent_location_updated = str(node_c_code_original_start_index) + "#" + str(
                                node_c_code_original_end_index)
                            parent_code_original = code_updated.split("&&&&&")[most_close_res_index]
                            one_node['location_updated'] = parent_location_updated
                            one_node['code_updated'] = parent_code_original
                    else:
                        if one_node_type.strip()=='Function':
                            original_code, parent_code_original = extract_by_offset(c_file_path, int(line), int(col), int(loc_3), int(end))
                            # node_c_code_original_start_index = original_code.index(parent_code_original)
                            # node_c_code_original_end_index = node_c_code_original_start_index + len(
                            #     parent_code_original)
                            node_c_code_original_start_index = 0
                            node_c_code_original_end_index = len(original_code)

                            parent_location_updated = str(node_c_code_original_start_index) + "#" + str(
                                node_c_code_original_end_index)
                            one_node['location_updated'] = parent_location_updated
                            one_node['code_updated'] = original_code

        joern_nodes_updated.append(one_node)


    return joern_nodes_updated




def update_joern_nodes_location_step_2(joern_nodes):
    joern_nodes_updated = []
    for i in range(len(joern_nodes)-1,-1,-1):

        one_node = joern_nodes[i]
        # print("one_node---:   ", one_node)
        one_node_location = one_node['location'].strip()
        # print("before clean:", one_node['code'])
        one_node_code = clean_joern_code(one_node['code'])
        # print("after clean:", one_node_code)
        one_node_type = one_node['type']

        one_node_location_updated = one_node['location_updated'].strip()
        one_node_code_updated = one_node['code_updated']

        if len(one_node_code.strip()) >= 0 and one_node_type != 'Symbol':
            if "&&&&&" in  one_node_location_updated:
                start_in_original, end_in_original = one_node_location_updated.split("#")
                start_list = start_in_original.split('&&&&&')
                end_list = end_in_original.split('&&&&&')
                start_list = [int(item) for item in start_list]
                end_list = [int(item) for item in end_list]

                for j in range(i - 1, -1, -1):
                    another_node = joern_nodes[j]
                    another_node_location = another_node['location'].strip()
                    another_node_code = another_node['code']
                    another_node_type = another_node['type']
                    another_node_location_updated = another_node['location_updated'].strip()
                    another_node_code_updated = another_node['code_updated']

                    if len(another_node_location) != 0 and another_node_location_updated !="":
                        parent_code = another_node_code
                        parent_location_updated = another_node_location_updated
                        parent_code_updated = another_node['code_updated']
                        parent_location_updated_start = int(parent_location_updated.split("#")[0])
                        parent_location_updated_end = int(parent_location_updated.split("#")[1])

                        for ii in range(len(start_list)):
                            one_start = start_list[ii]
                            one_end = end_list[ii]
                            if one_start >= parent_location_updated_start and one_end <= parent_location_updated_end:
                                one_node['location_updated'] = str(one_start)+"#"+ str(one_end)
                                one_node['code_updated'] = one_node_code_updated.split("&&&&&")[0]
                                break
                        break
        joern_nodes_updated.append(one_node)

    joern_nodes_updated = list(reversed(joern_nodes_updated))
    joern_nodes_updated = update_joern_nodes_location_step_2_patch(joern_nodes_updated)
    return joern_nodes_updated




def update_joern_nodes_location_step_2_patch(joern_nodes):
    joern_nodes_updated = []
    for i in range(len(joern_nodes)-1,-1,-1):

        one_node = joern_nodes[i]
        # print("one_node---:   ", one_node)
        one_node_location = one_node['location'].strip()
        # print("before clean:", one_node['code'])
        one_node_code = clean_joern_code(one_node['code'])
        # print("after clean:", one_node_code)
        one_node_type = one_node['type']

        one_node_location_updated = one_node['location_updated'].strip()
        one_node_code_updated = one_node['code_updated']

        if len(one_node_code.strip()) >= 0 and one_node_type != 'Symbol':
            if "&&&&&" in  one_node_location_updated:
                start_in_original, end_in_original = one_node_location_updated.split("#")
                start_list = start_in_original.split('&&&&&')
                end_list = end_in_original.split('&&&&&')
                start_list = [int(item) for item in start_list]
                end_list = [int(item) for item in end_list]

                for j in range(i - 1, -1, -1):
                    another_node = joern_nodes[j]
                    another_node_location = another_node['location'].strip()
                    another_node_code = another_node['code']
                    another_node_type = another_node['type']
                    another_node_location_updated = another_node['location_updated'].strip()
                    another_node_code_updated = another_node['code_updated']

                    if len(another_node_location) != 0 and another_node_location_updated !="":
                        parent_code = another_node_code
                        parent_location_updated = another_node_location_updated
                        parent_code_updated = another_node['code_updated']
                        parent_location_updated_start = int(parent_location_updated.split("#")[0])
                        parent_location_updated_end = int(parent_location_updated.split("#")[1])

                        start_list = sorted(start_list)
                        end_list = sorted(end_list)
                        potential_distance = []
                        select_index = 0
                        for ii in range(len(start_list)):
                            one_start = start_list[ii]
                            one_end = end_list[ii]
                            distance = one_start-parent_location_updated_start
                            potential_distance.append(distance)
                            if distance<0:
                                continue
                            else:
                                select_index = ii
                                break

                        one_node['location_updated'] = str(start_list[select_index]) + "#" + str(end_list[select_index])
                        one_node['code_updated'] = one_node_code_updated.split("&&&&&")[0]

                        break
        joern_nodes_updated.append(one_node)

    joern_nodes_updated = list(reversed(joern_nodes_updated))
    return joern_nodes_updated

def update_joern_nodes_location_step_3(joern_nodes):
    joern_nodes_updated = []
    for i in range(len(joern_nodes)-1,-1,-1):

        one_node = joern_nodes[i]
        one_node_location = one_node['location']
        one_node_code = one_node['code']
        one_node_type = one_node['type']
        location_updated = one_node['location_updated']


        if one_node_type == 'Symbol' and "&&&&&" in location_updated:

            for j in range(i-1,-1,-1):
                another_node = joern_nodes[j]
                another_node_location = another_node['location']
                another_node_code = another_node['code']
                another_node_type = another_node['type']

                if another_node_type != 'Symbol':
                    if one_node_code == another_node_code:
                        one_node['location_updated'] = another_node['location_updated']
                        one_node['code_updated'] = another_node['code_updated']
                        break

        joern_nodes_updated.append(one_node)


    joern_nodes_updated = list(reversed(joern_nodes_updated))
    return joern_nodes_updated


def update_joern_nodes_location_step_0(joern_nodes, c_file_path):

    with open(c_file_path, "r") as f:
        original_code = f.read()

    joern_nodes_updated = []
    for i in range(len(joern_nodes)):
        # print("i:", i)
        one_node = joern_nodes[i]
        # print("one_node---:   ", one_node)
        one_node_location = one_node['location']
        # print("before clean:", one_node['code'])
        one_node_code = clean_joern_code(one_node['code'])
        # print("after clean:", one_node_code)
        one_node['location_updated'] = ""
        one_node['code_updated'] = ""

        if len(one_node_code)!= 0:
            parent_code_original, node_c_code_original_start_index, node_c_code_original_end_index = find_index_from_parent_code(
                original_code, one_node_code)

            if parent_code_original != "":
                one_node['location_updated'] = str(node_c_code_original_start_index) + "#" + str(node_c_code_original_end_index)
                one_node['code_updated'] = parent_code_original

        joern_nodes_updated.append(one_node)

    return joern_nodes_updated




def update_joern_nodes_location(joern_nodes,c_file_path ):
    # joern_nodes = read_csv(nodes_path)

    joern_nodes_updated = update_joern_nodes_location_step_0(joern_nodes, c_file_path)
    joern_nodes_updated = update_joern_nodes_location_step_1(joern_nodes_updated, c_file_path)
    joern_nodes_updated = update_joern_nodes_location_step_2(joern_nodes_updated)
    joern_nodes_updated = update_joern_nodes_location_step_3(joern_nodes_updated)

    return joern_nodes_updated


if __name__ == '__main__':

    t_test_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/test/chrome/140100/140100_func_before_target_0.c/nodes.csv'
    joern_nodes = read_csv(t_test_path)
    joern_nodes = pd.DataFrame(joern_nodes)
    joern_nodes.to_excel('/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/test/chrome/140100/140100_func_before_target_0.c/nodes_df_140100.xlsx', index=False)
    exit()

    # test update_joern_nodes_location_step_2
    t_test_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/train/chrome/140744/140744_func_before_target_0.c/nodes.csv'
    c_file_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/train/chrome/140744/140744_func_before_target_0.c'

    t_test_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/test/linux/181346/181346_func_before_target_1.c/nodes.csv'
    c_file_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/test/linux/181346/181346_func_before_target_1.c'
    joern_nodes = read_csv(t_test_path)
    joern_nodes_updated = update_joern_nodes_location_step_0(joern_nodes, c_file_path)
    joern_nodes_updated = update_joern_nodes_location_step_1(joern_nodes_updated, c_file_path)
    joern_nodes_updated = update_joern_nodes_location_step_2(joern_nodes_updated)
    joern_nodes_updated = update_joern_nodes_location_step_3(joern_nodes_updated)


    exit()


    t_test_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/train/chrome/140744/140744_func_before_target_0.c/nodes.csv'
    c_file_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/train/chrome/140744/140744_func_before_target_0.c'

    joern_nodes = read_csv(t_test_path)

    joern_nodes_updated = update_joern_nodes_location_step_0(joern_nodes, c_file_path)
    joern_nodes_updated = update_joern_nodes_location_step_1(joern_nodes_updated, c_file_path)
    joern_nodes_updated = update_joern_nodes_location_step_2(joern_nodes_updated)
    joern_nodes_updated = update_joern_nodes_location_step_3(joern_nodes_updated)

    for item in joern_nodes_updated:
        print(item)

    joern_nodes_updated = pd.DataFrame(joern_nodes_updated)
    print(joern_nodes_updated)
    print(joern_nodes_updated.columns.values)
    joern_nodes_updated = joern_nodes_updated[['command', 'key', 'code', 'location', 'location_updated', 'code_updated']]  # 'location_relative'
    joern_nodes_updated.to_excel('/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/140744.xlsx', index=False)
    exit()









    t_test_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/train/chrome/140744/140744_func_before_target_0.c/nodes.csv'
    select_real_code_from_nodes_csv(t_test_path)
    directory = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code'
    csv_files, csv_dict, v_instance_dict = collect_csv_files(directory)

    for k, v in csv_dict.items():
        for file in v:
            node_data = read_csv(file)
            if len(node_data)>=0:
                node_df =pd.DataFrame(node_data)
                node_df.to_excel(file + ".xlsx", index=False)

    # 保存原始 C++ 函数到 test_func.c 文件
    sample_code = '''void GLES2DecoderImpl::DoLinkProgram(GLuint program_id) {
      TRACE_EVENT0("gpu", "GLES2DecoderImpl::DoLinkProgram");
      Program* program = GetProgramInfoNotShader(
          program_id, "glLinkProgram");
      if (!program) {
        return;
      }

      LogClientServiceForInfo(program, program_id, "glLinkProgram");
      if (program->Link(shader_manager(),
                        workarounds().count_all_in_varyings_packing ?
                            Program::kCountAll : Program::kCountOnlyStaticallyUsed,
                        shader_cache_callback_)) {
        if (program == state_.current_program.get()) {
          if (workarounds().use_current_program_after_successful_link)
            glUseProgram(program->service_id());
          if (workarounds().clear_uniforms_before_first_program_use)
            program_manager()->ClearUniforms(program);
        }
      }

      ExitCommandProcessingEarly();
    };'''

    # file_path = "/mnt/data/test_func.c"
    # with open(file_path, "w") as f:
    #     f.write(sample_code)


    # 定义偏移提取函数
    def extract_by_offset(filename, start_line, start_column, end_offset):
        with open(filename, "r") as f:
            full_text = f.read()

        lines = full_text.splitlines(keepends=True)
        start_index = sum(len(lines[i]) for i in range(start_line - 1)) + start_column
        return full_text[start_index:end_offset]


    # 测试三个 location 字段
    locations = [
        (1, 0, 851),  # 整个函数体
        (2, 2, 114),  # TRACE_EVENT0 行
        (6, 4, 226),  # return;
        (3, 2, 196),  # return;
        (13, 45, 470),  # return;
        (15, 10, 644),  # return;
    ]

    file_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/train/chrome/140744/140744_func_before_target_0.c'
    # 提取并展示结果
    results = [extract_by_offset(file_path, line, col, end) for (line, col, end) in locations]
    for item in results:
        print("item:\n", item)
