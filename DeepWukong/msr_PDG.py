import subprocess
import os
import time

# 日志文件路径
log_file_path = "logs/log.log"
error_folder = "data/msr/error"
error_skip_folder = "data/msr/error_skip"
already_folder_PDG = "data/msr/already_PDG"

# 确保 error 目录存在
os.makedirs(error_folder, exist_ok=True)

def run_joern_parse():
    """运行 joern-parse.py，并将输出重定向到 logs/log.log"""
    command = "python src/joern/joern-parse.py -c /scratch/c00590656/vulnerability/DeepWukong/configs/msr.yaml > logs/log.log 2>&1"
    result = subprocess.run(command, shell=True)
    return result.returncode  # 获取执行状态

def read_c_files_from_log():
    """读取 logs/log.log 并提取 .c 结尾的文件路径"""
    c_files = []
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            for line in log_file:
                line = line.strip()
                if line.endswith(".c") and os.path.exists(line):  # 确保文件存在
                    c_files.append(line)
                if 'Error' in line and 'skipping' in line:
                    c_files.append(line)
    return c_files

def process_c_files(c_files):
    """处理 c 文件：
    - 最后一个 .c 文件移动到 data/msr/error
    - 其他的 .c 文件删除
    """
    if not c_files:
        return

    for i, file_path in enumerate(c_files):

        if 'skipping' in file_path and 'Erro' in file_path:
            continue

        if i != len(c_files) - 1:
            line_i_plus_1_content = c_files[i + 1]
        else:
            line_i_plus_1_content = ""
        try:

            if 'skipping' in line_i_plus_1_content and 'Erro' in line_i_plus_1_content:

                destination = os.path.join(error_skip_folder, os.path.basename(file_path))
                shutil.move(file_path, destination)  # 移动文件
                print(f"skipping Erro Moved: {file_path} -> {destination}")
            else:
                if i == len(c_files) - 1:  # 最后一个文件
                    destination = os.path.join(error_folder, os.path.basename(file_path))
                    os.rename(file_path, destination)  # 移动文件
                    print(f"Moved: {file_path} -> {destination}")
                else:  # 其他文件
                    # os.remove(file_path)  # 删除文件
                    # print(f"Deleted: {file_path}")
                    destination = os.path.join(already_folder_PDG, os.path.basename(file_path))
                    os.rename(file_path, destination)  # 移动文件
                    print(f"Moved: {file_path} -> {destination}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


import os
import shutil


# def move_files(source_dir, target_dir):
#     """
#     Move all files and subdirectories from source_dir to target_dir.
#
#     Args:
#         source_dir (str): The source directory path.
#         target_dir (str): The target directory path.
#     """
#     # Ensure source directory exists
#     if not os.path.exists(source_dir):
#         print(f"Error: Source directory '{source_dir}' does not exist.")
#         return
#
#     # Create target directory if it doesn't exist
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#         print(f"Created target directory: {target_dir}")
#
#     # Iterate over all items in the source directory
#     for item in os.listdir(source_dir):
#         source_path = os.path.join(source_dir, item)
#         target_path = os.path.join(target_dir, item)
#
#         # try:
#         if os.path.isfile(source_path):
#             # Move file
#             os.rename(source_path, target_path)
#             print(f"Moved file: {source_path} -> {target_path}")
#         elif os.path.isdir(source_path):
#             # For directories, recursively move contents
#             for root, dirs, files in os.walk(source_path, topdown=False):
#                 # Move all files in current directory
#                 for file in files:
#                     file_source = os.path.join(root, file)
#                     # Calculate relative path and create target path
#                     rel_path = os.path.relpath(file_source, source_dir)
#                     file_target = os.path.join(target_dir, rel_path)
#                     # Create subdirectory structure if it doesn't exist
#                     os.makedirs(os.path.dirname(file_target), exist_ok=True)
#                     shutil.move(file_source, file_target)
#                     print(f"Moved file: {file_source} -> {file_target}")
#
#                 # After moving all contents, remove empty directories
#                 for root, dirs, files in os.walk(source_path, topdown=False):
#                     if not os.listdir(root):  # If directory is empty
#                         os.rmdir(root)
#                         print(f"Removed empty directory: {root}")
#
#
#     # Verify if source directory is empty after moving
#     if not os.listdir(source_dir):
#         print(f"All items have been successfully moved from {source_dir}.")
#     else:
#         print(f"Warning: Some items remain in {source_dir}. Check for errors above.")
#


import os


def remove_empty_dirs(root_dir):
    """
    Recursively remove all empty directories under root_dir.

    Args:
        root_dir (str): The root directory to start from.
    """
    # Ensure the root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    # Use os.walk with topdown=False to process subdirectories first
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Skip the root directory itself
        if dirpath == root_dir:
            continue

        # Check if the directory is empty
        try:
            if not os.listdir(dirpath):  # If no files or subdirs
                os.rmdir(dirpath)
                print(f"Removed empty directory: {dirpath}")
        except OSError as e:
            print(f"Error removing {dirpath}: {str(e)}")

    # Check if root_dir itself is empty after cleaning subdirs
    try:
        if not os.listdir(root_dir):
            os.rmdir(root_dir)
            print(f"Removed empty root directory: {root_dir}")
    except OSError as e:
        print(f"Error removing root directory {root_dir}: {str(e)}")






def move_files(source_dir, target_dir):
    """
    Move all contents from source_dir to target_dir, merging with existing content.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created target directory: {target_dir}")

    # 移动 source_dir 中的所有内容到 target_dir
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)

        if os.path.isfile(source_path):
            if os.path.exists(target_path):
                print(f"Overwriting file {target_path}")
            shutil.move(source_path, target_path)
            print(f"Moved file: {source_path} -> {target_path}")
        elif os.path.isdir(source_path):
            if os.path.exists(target_path):
                print(f"Merging directory {target_path}")
                move_files(source_path, target_path)  # 递归合并
            else:
                shutil.move(source_path, target_path)
                print(f"Moved directory: {source_path} -> {target_path}")



    if not os.path.exists(source_dir):
        os.mkdir(source_dir)




def main():
    """主循环，不断执行解析，直到 logs/log.log 里没有 .c 文件路径"""
    while True:
        return_code = run_joern_parse()
        print("return_code:\n", return_code)
        if return_code == 0:  # 如果命令失败，解析日志
            print("Joern parse failed. Checking log for C files...")

            c_files = read_c_files_from_log()
            if not c_files:
                print("No .c files found in log. Exiting.")
                break  # 结束循环

            process_c_files(c_files)

            source_directory = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv"
            target_directory = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv_already"

            move_files(source_directory, target_directory)

            root_directory = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code"
            remove_empty_dirs(root_directory)

        else:
            print("Joern parse completed successfully. Exiting.")
            break  # 结束循环

        time.sleep(2)  # 等待 2 秒，防止 CPU 过载

if __name__ == "__main__":
    # First, run PYTHONPATH="."
    main()

    # source_directory = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv"
    # target_directory = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/csv_already"
    # #
    # move_files(source_directory, target_directory)
    # # 指定目录
    # root_directory = "/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code"
    # remove_empty_dirs(root_directory)
