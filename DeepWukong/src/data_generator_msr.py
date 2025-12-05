import xml.etree.ElementTree as ET
import networkx as nx
from typing import List, Set, Tuple, Dict
from os.path import join, exists
from argparse import ArgumentParser
import os
from tqdm import tqdm
from typing import cast
import dataclasses
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Manager, Pool, Queue
import functools
import pandas as pd
import math
USE_CPU = cpu_count()



def get_vulnerable_lines():
    already_id = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_Intersection_Visual_LineVul_Map'
    already_id = os.listdir(already_id)


    c_root = '/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset'
    files_list = ['test.csv', 'val.csv', 'train.csv']

    vulnerable_lines = {}
    for file in files_list:
        file_path = os.path.join(c_root, file)
        file_df = pd.read_csv(file_path)
        print(file_df.columns.values)
        for ii, row in file_df.iterrows():
            index = row['index']
            vulnerable_lines[int(index)] = row['flaw_line_index']
            if str(index) in already_id:
                print("\n\n******************\nindex:\n", row['index'])
                print("\ntarget:\n", row['target'])
                print("\ncommit_id:\n", row['commit_id'])
                print("\nfunc_before:\n", row['func_before'])
                print("\nflaw_line_index:\n", row['flaw_line_index'])
                print("\nflaw_line:\n", row['flaw_line'])
    return vulnerable_lines


vulnerable_lines_dict = get_vulnerable_lines()

def extract_line_number(idx: int, nodes: List) -> int:
    """
    return the line number of node index

    Args:
        idx (int): node index
        nodes (List)
    Returns: line number of node idx
    """
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except Exception as e:
                    print(e)
                    pass
        idx -= 1
    return -1


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


def extract_nodes_with_location_info(nodes):
    """
    Will return an array identifying the indices of those nodes in nodes array
    another array identifying the node_id of those nodes
    another array indicating the line numbers
    all 3 return arrays should have same length indicating 1-to-1 matching.
    
    """

    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    for node_index, node in enumerate(nodes):
        assert isinstance(node, dict)
        if 'location' in node.keys():
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
    return node_indices, node_ids, line_numbers, node_id_to_line_number


def build_PDG(code_path: str, sensi_api_path: str,
              source_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    """
    build program dependence graph from code

    Args:
        code_path (str): source code root path
        sensi_api_path (str): path to sensitive apis
        source_path (str): source file path

    Returns: (PDG, key line map)
    """
    print("code_path:", code_path)
    nodes_path = join(code_path, "nodes.csv")
    edges_path = join(code_path, "edges.csv")
    print("nodes_path:", nodes_path)
    print("edges_path:", edges_path)

    # 用斜杠分割路径
    parts = nodes_path.split("/")

    file_id = int(parts[-3])
    file_id_flaw_line_index = vulnerable_lines_dict[file_id]
    if ',' in str(file_id_flaw_line_index):
        file_id_flaw_line_index = str(file_id_flaw_line_index).split(",")
    else:
        file_id_flaw_line_index = [file_id_flaw_line_index]

    if len(file_id_flaw_line_index) >= 1:
        file_id_flaw_line_index = [int(item)+1 for item in file_id_flaw_line_index if not (item is None or isinstance(item, float) and math.isnan(item))]


    assert exists(sensi_api_path), f"{sensi_api_path} not exists!"
    with open(sensi_api_path, "r", encoding="utf-8") as f:
        sensi_api_set = set([api.strip() for api in f.read().split(",")])
    if not exists(nodes_path) or not exists(edges_path):
        return None, None
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    print("nodes_path loaded:", nodes_path)
    print("edges_path loaded:", edges_path)
    call_lines = set()
    array_lines = set()
    ptr_lines = set()
    arithmatic_lines = set()
    vulnerable_lines = set()
    if len(nodes) == 0:
        return None, None
    print("nodes_path loaded length >=0", nodes_path)
    print("edges_path loaded length >=0", edges_path)
    for node_idx, node in enumerate(nodes):
        ntype = node['type'].strip()
        nkey = int(node['key'])
        # print("nkey:", nkey)
        # print("node_idx:", node_idx)
        if ntype == 'CallExpression':
            function_name = nodes[node_idx + 1]['code']
            if function_name is None or function_name.strip() == '':
                continue
            if function_name.strip() in sensi_api_set:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    call_lines.add(line_no)
                    if line_no in file_id_flaw_line_index:
                        vulnerable_lines.add(line_no)
        elif ntype == 'ArrayIndexing':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                array_lines.add(line_no)
                if line_no in file_id_flaw_line_index:
                    vulnerable_lines.add(line_no)
        elif ntype == 'PtrMemberAccess':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                ptr_lines.add(line_no)
                if line_no in file_id_flaw_line_index:
                    vulnerable_lines.add(line_no)
        elif node['operator'].strip() in ['+', '-', '*', '/']:
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                arithmatic_lines.add(line_no)
                if line_no in file_id_flaw_line_index:
                    vulnerable_lines.add(line_no)
        else:
            try:
                line_no = extract_line_number(node_idx, nodes)
                if line_no in file_id_flaw_line_index:
                    vulnerable_lines.add(line_no)
            except Exception as e:
                print(e)

    if len(vulnerable_lines)>= 0:
        print("vulnerable_lines:\n", vulnerable_lines)

    PDG = nx.DiGraph(file_paths=[source_path])
    control_edges, data_edges = list(), list()
    node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(
        nodes)
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_ln.keys(
            ) or end_node_id not in node_id_to_ln.keys():
                continue
            start_ln = node_id_to_ln[start_node_id]
            end_ln = node_id_to_ln[end_node_id]
            if edge_type == 'CONTROLS':  # Control
                control_edges.append((start_ln, end_ln, {"c/d": "c"}))
            if edge_type == 'REACHES':  # Data
                data_edges.append((start_ln, end_ln, {"c/d": "d"}))
    PDG.add_edges_from(control_edges)
    PDG.add_edges_from(data_edges)
    return PDG, {
        "call": call_lines,
        "array": array_lines,
        "ptr": ptr_lines,
        "arith": arithmatic_lines,
        "vulnerable": vulnerable_lines,
    }


def build_XFG(PDG: nx.DiGraph, key_line_map: Dict[str, Set[int]],
              vul_lines: Set[int] = None) -> Dict[str, List[nx.DiGraph]]:
    """
    build XFGs
    Args:
        PDG (nx.DiGraph): program dependence graph
        key_line_map (Dict[str, Set[int]]): key lines
    Returns: XFG map
    """
    if PDG is None or key_line_map is None:
        return None
    # ct0, ct1 = 0, 0
    res = {"call": [], "array": [], "ptr": [], "arith": [], "vulnerable":[]}
    for key in ["call", "array", "ptr", "arith", "vulnerable"]:
        for ln in key_line_map[key]:
            sliced_lines = set()

            # backward traversal
            bqueue = list()
            visited = set()
            bqueue.append(ln)
            visited.add(ln)
            while bqueue:
                fro = bqueue.pop(0)
                sliced_lines.add(fro)
                if fro in PDG._pred:
                    for pred in PDG._pred[fro]:
                        if pred not in visited:
                            visited.add(pred)
                            bqueue.append(pred)

            # forward traversal
            fqueue = list()
            visited = set()
            fqueue.append(ln)
            visited.add(ln)
            while fqueue:
                fro = fqueue.pop(0)
                sliced_lines.add(fro)
                if fro in PDG._succ:
                    for succ in PDG._succ[fro]:
                        if succ not in visited:
                            visited.add(succ)
                            fqueue.append(succ)
            if len(sliced_lines) != 0:
                XFG = PDG.subgraph(list(sliced_lines)).copy()
                XFG.graph["key_line"] = ln
                if vul_lines is not None:
                    if len(sliced_lines.intersection(vul_lines)) != 0:
                        XFG.graph["label"] = 1
                        # ct1 += 1
                    else:
                        XFG.graph["label"] = 0
                        # ct0 += 1
                else:
                    XFG.graph["label"] = "UNK"
                res[key].append(XFG)
        # print("ct1:", ct1)
        # print("ct0:", ct0)

    return res


def getCodeIDtoPathDict(testcases: List,
                        sourceDir: str) -> Dict[str, Dict[str, Set[int]]]:
    '''build code testcaseid to path map

    use the manifest.xml. build {testcaseid:{filePath:set(vul lines)}}
    filePath use relevant path, e.g., CWE119/cve/source-code/project_commit/...
    :param testcases:
    :return: {testcaseid:{filePath:set(vul lines)}}
    '''
    codeIDtoPath: Dict[str, Dict[str, Set[int]]] = {}
    for testcase in testcases:
        files = testcase.findall("file")
        testcaseid = testcase.attrib["id"]
        codeIDtoPath[testcaseid] = dict()

        for file in files:
            path = file.attrib["path"]
            flaws = file.findall("flaw")
            mixeds = file.findall("mixed")
            fix = file.findall("fix")
            # print(mixeds)
            VulLine = set()
            if (flaws != [] or mixeds != [] or fix != []):
                # targetFilePath = path
                if (flaws != []):
                    for flaw in flaws:
                        VulLine.add(int(flaw.attrib["line"]))
                if (mixeds != []):
                    for mixed in mixeds:
                        VulLine.add(int(mixed.attrib["line"]))

            codeIDtoPath[testcaseid][path] = VulLine



    for k, v in codeIDtoPath.items():
        v_items_cnt = len(v.keys())
        if v_items_cnt == 2:
            non_v = 0
            v_v = 0
            non_v_name = ""
            for item in v.keys():
                if '_0.c' in item:
                    non_v = 1
                    non_v_name = item
                if '_1.c' in item:
                    v_v = 1
            if v_v == 1 and non_v == 1:
                del codeIDtoPath[k][non_v_name]


    return codeIDtoPath


def dump_XFG(res: Dict[str, List[nx.DiGraph]], out_root_path: str,
             testcaseid: str):
    """
    dump XFG to file

    Args:
        res: XFGs
        out_root_path: output root path
        testcaseid: testcase id
    Returns:
    """
    if res is None:
        return
    testcase_out_root_path = join(out_root_path, testcaseid)
    if not exists(testcase_out_root_path):
        os.makedirs(testcase_out_root_path)
    for k in res:
        k_root_path = join(testcase_out_root_path, k)
        if not exists(k_root_path):
            os.makedirs(k_root_path)
        for XFG in res[k]:
            out_path = join(k_root_path, f"{XFG.graph['key_line']}.xfg.pkl")
            print("out_path:", out_path)
            print("XFG:", XFG)
            nx.write_gpickle(XFG, out_path)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    return arg_parser


@dataclasses.dataclass
class QueueMessage:
    XFG_res: Dict
    out_root_path: str
    testcaseid: str
    is_finished: bool = False


def handle_queue_message(queue: Queue):
    """

    Args:
        queue:

    Returns:

    """
    xfg_ct = 0
    while True:
        message: QueueMessage = queue.get()
        if message.is_finished:
            break
        if message.XFG_res is not None:
            dump_XFG(message.XFG_res, message.out_root_path, message.testcaseid)
            for k in message.XFG_res:
                xfg_ct += len(message.XFG_res[k])
    return xfg_ct


def process_parallel(testcase: ET.Element, queue: Queue, doneIDs: Set, codeIDtoPath: Dict, cwe_root: str,
                     source_root_path: str,
                     out_root_path: str):
    """

    Args:
        testcase:
        doneIDs:
        codeIDtoPath:
        cwe_root:
        source_root_path:
        out_root_path:

    Returns:

    """
    print("2222")
    testcaseid = testcase.attrib["id"]
    # testcaseid = '179655'
    # for testcaseid in ['179655', '180688', '181013', '183528', '183712', '184827', '185116', '185352', '186157', '186623', '186678', '187878', '187989']:
    # for testcaseid in ['178356', '178640', '178184', '181067', '181054', '181950', '178625', '178105', '181701', '181132', '181028', '181107', '178458']:

    if testcaseid in codeIDtoPath:
        file_map = codeIDtoPath[testcaseid]

        for file_path in file_map:
            # if testcaseid in ['179655', '180688', '181013', '183528', '183712', '184827', '185116', '185352', '186157', '186623', '186678', '187878', '187989']:
            # if testcaseid in ['178356', '178640', '178184', '181067', '181054', '181950', '178625', '178105', '181701', '181132', '181028', '181107', '178458']:
            #
            #     print("testcaseid:", testcaseid)
            #     pass
            # else:
            #     continue
            print("\n\n********")
            print("source_root_path:", source_root_path)
            print("file_path:", file_path)
            vul_lines = file_map[file_path]
            csv_path = join(cwe_root, "csv", os.path.abspath(source_root_path)[1:],
                            file_path)
            print("csv_path 360:", csv_path)
            source_path = join(source_root_path, file_path)
            print("source_path 362:", csv_path)
            print("vul_lines:", vul_lines)
            PDG, key_line_map = build_PDG(csv_path, "/scratch/c00590656/vulnerability/DeepWukong/data/sensiAPI.txt",
                                          source_path)
            res = build_XFG(PDG, key_line_map, vul_lines)
            queue.put(QueueMessage(res, out_root_path, testcaseid))
            print("out_root_path ******:", out_root_path)
            dump_XFG(res, out_root_path, testcaseid)
    return testcaseid


def generate(config_path: str):
    config_path = '/scratch/c00590656/vulnerability/DeepWukong/configs/msr.yaml'
    config = cast(DictConfig, OmegaConf.load(config_path))
    root = config.data_folder
    cweid = config.dataset.name
    cwe_root = join(root, cweid)
    source_root_path = join(cwe_root, "source-code")
    out_root_path = join(cwe_root, "XFG")
    xml_path = join(source_root_path, "manifest.xml")

    cwe_root = '/scratch/c00590656/vulnerability/DeepWukong/data/msr'
    source_root_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code'
    xml_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/xml/manifest.xml'

    print("root:", root)
    print("cweid:", cweid)
    print("out_root_path:", out_root_path)
    print("source_root_path:", source_root_path)
    print("xml_path:", xml_path)


    tree = ET.ElementTree(file=xml_path)
    testcases = tree.findall("testcase")
    codeIDtoPath = getCodeIDtoPathDict(testcases, source_root_path)

    if not exists(out_root_path):
        os.makedirs(out_root_path)
    if not exists(join(cwe_root, "doneID.txt")):
        os.system("touch {}".format(join(cwe_root, "doneID.txt")))
    with open(join(cwe_root, "doneID.txt"), "r", encoding="utf-8") as f:
        doneIDs = set(f.read().split(","))
    print("1111111")
    testcase_len = len(testcases)
    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        pool = Pool(USE_CPU)
        xfg_ct = pool.apply_async(handle_queue_message, (message_queue,))
        process_func = functools.partial(process_parallel, queue=message_queue, doneIDs=doneIDs,
                                         codeIDtoPath=codeIDtoPath,
                                         cwe_root=cwe_root, source_root_path=source_root_path,
                                         out_root_path=out_root_path)
        testcaseids_done: List = [
            testcaseid
            for testcaseid in tqdm(
                pool.imap_unordered(process_func, testcases),
                desc=f"testcases: ",
                total=testcase_len,
            )
        ]

        message_queue.put(QueueMessage(None, None, None, True))
        pool.close()
        pool.join()
    print(f"total {xfg_ct.get()} XFGs!")
    for testcaseid in testcaseids_done:
        with open(join(cwe_root, "doneID.txt"), 'a',
                  encoding="utf-8") as f:
            f.write(str(testcaseid))
            f.write(",")


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    generate(__args.config)
