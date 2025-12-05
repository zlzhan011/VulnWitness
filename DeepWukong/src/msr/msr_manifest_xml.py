import os
import hashlib
from xml.etree import ElementTree as ET
from datetime import datetime
import pandas as pd
# 计算文件的 SHA-1 校验和
def calculate_checksum(file_path):
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        sha1.update(f.read())
    return sha1.hexdigest()



def read_msr_dataset():
    c_root = '/scratch/c00590656/vulnerability/LineVul/data/big-vul_dataset'
    # c_output = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/'
    ttv = []
    files = ['val.csv', 'test.csv', 'train.csv']
    # files = ['val.csv']
    for file in files:
        file_path = os.path.join(c_root, file)
        file_df = pd.read_csv(file_path)
        ttv.append(file_df)

    ttv = pd.concat(ttv)
    print(ttv.shape)
    print(ttv.columns.values)

    ttv_dict = {}
    for i, row in ttv.iterrows():
        index = row['index']
        CWE_ID = row['CWE ID']
        CVE_ID = row['CVE ID']
        target = row['target']
        flaw_line_index = row['flaw_line_index']
        # 处理 flaw_line_index
        if pd.isna(flaw_line_index):  # 如果是 nan
            flaw_line_index = []
        else:  # 如果是字符串（如 "1,2,3,4,5"）
            flaw_line_index = [str(x) for x in str(flaw_line_index).split(',')]


        ttv_dict[str(index)] = {"CWE_ID":str(CWE_ID),
                           "CVE_ID":str(CVE_ID),
                           "target":str(target),
                           "flaw_line_index":flaw_line_index}


    return ttv_dict



def get_manifest_xml():
    ttv_dict = read_msr_dataset()
    # exit()

    # 文件目录和基本信息
    # base_dir = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/already_PDG"  # 替换为你的文件目录
    base_dir = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code'
    output_dir = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/xml'
    output_file = os.path.join(output_dir,  "manifest.xml")
    author = "Lei"
    submission_date = datetime.now().strftime("%Y-%m-%d")
    testsuite_id = "99"
    start_id = 10001

    # 创建 XML 结构
    container = ET.Element("container")

    # 遍历目录中的文件
    testcase_files = {}  # 按测试用例分组（假设文件名有规律）
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".c") or file.endswith(".h"):
                # 假设文件名格式如 "CWE119_test_01.c" 表示一个测试用例
                testcase_id = file.split("_")[0]  # 提取测试用例标识
                if testcase_id not in testcase_files:
                    testcase_files[testcase_id] = []
                testcase_files[testcase_id].append(os.path.join(root, file))

    # 为每个测试用例生成 <testcase>
    for i, (testcase_id, files) in enumerate(testcase_files.items(), start=start_id):
        testcase = ET.SubElement(container, "testcase", {
            "id": str(testcase_id),
            "index": str(testcase_id),
            "target": ttv_dict[testcase_id]["target"],
            "type": "Source Code",
            "status": "Accepted",
            "submissionDate": submission_date,
            "language": "C",
            "author": author,
            "numberOfFiles": str(len(files)),
            # "testsuiteid": testsuite_id
        })

        # 添加描述（这里需要你手动调整）
        description = ET.SubElement(testcase, "description")
        description.text = f"CWE: 119 Buffer Overflow<br/> BadSource: User input<br/> GoodSource: Fixed value<br/> Sinks:<br/> GoodSink: Bounds check<br/> BadSink: No bounds check<br/> Flow Variant: 01 Basic flow"
        description.text = 'MSR dataset'
        # 添加文件
        for file_path in files:
            file_size = os.path.getsize(file_path)
            checksum = calculate_checksum(file_path)
            relative_path = os.path.relpath(file_path, base_dir)
            file_elem = ET.SubElement(testcase, "file", {
                "path": relative_path,
                "language": "C",
                "size": str(file_size),
                "checksum": checksum
            })
            # 如果是主测试文件，添加漏洞位置（假设第 25 行）
            if file_path.endswith(".c"):
                # 获取该文件的漏洞行信息，默认为空列表
                flaws = ttv_dict[testcase_id]["flaw_line_index"]
                # print("flaws:", flaws)
                if len(flaws) > 0:
                    # 循环添加每个漏洞行
                    for line in flaws:
                        flaw_name = ttv_dict[testcase_id]["CWE_ID"]
                        ET.SubElement(file_elem, "flaw", {"line": str(line), "name": flaw_name})

    # 保存到文件
    tree = ET.ElementTree(container)
    ET.indent(tree, space="  ")  # 美化格式
    # with open(output_file, "wb") as f:
    #     tree.write(f, encoding="utf-8", xml_declaration=True)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Generated {output_file}")


if __name__ == '__main__':
    get_manifest_xml()