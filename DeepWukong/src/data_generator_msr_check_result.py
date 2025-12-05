




import os
import networkx as nx



if __name__ == '__main__':
    c_root = '/scratch/c00590656/vulnerability/DeepWukong/src/data/msr/XFG'
    # for file_id in ['179655', '180688', '181013', '183528', '183712', '184827', '185116', '185352', '186157', '186623', '186678', '187878', '187989']:
    for file_id in ['179655', ]:

        file_id_dir = os.path.join(c_root, file_id)
        for slicing_method in os.listdir(file_id_dir):
            slicing_method_dir = os.path.join(file_id_dir, slicing_method)
            for file_name in os.listdir(slicing_method_dir):
                pkl_file = os.path.join(slicing_method_dir, file_name)
                if os.path.getsize(pkl_file) == 0:
                    print("\n\n\n**********error pkl_file:", pkl_file)
                    continue

                one_res = {}
                xfg = nx.read_gpickle(pkl_file)
                label_slicing = xfg.graph['label']
                num_nodes_slicing = len(xfg.nodes)
                num_nodes_func = 0

                one_res['xfg_file'] = pkl_file
                one_res['label_slicing'] = label_slicing
                one_res['num_nodes_slicing'] = num_nodes_slicing

                xfg_nodes = list(xfg.nodes)