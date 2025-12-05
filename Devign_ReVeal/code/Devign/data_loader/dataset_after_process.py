import re

import torch
from dgl import DGLGraph
from tqdm import tqdm

import dgl
from utils import load_default_identifiers, initialize_batch, debug
from pathlib import Path
import pandas as pd
import copy
import warnings
import jsonlines
import json
import itertools

class DataEntry:
    def __init__(self, _id, datset, num_nodes, features, edges, target, file_name):
        self._id = _id
        self.num_nodes = num_nodes
        self.target = target
        self.file_name = file_name
        self._file_name = file_name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.graph = DGLGraph()
        self.graph.add_nodes(
            self.num_nodes,
            data={'features': torch.FloatTensor(features)}
        )
        for s, _type, t in edges:
            etype_number = datset.get_edge_type_number(_type)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.graph.add_edges(s, t, data={'etype': torch.LongTensor([etype_number])})


class DataSet:
    def __init__(self, all_src, batch_size=32, n_ident=None, g_ident=None, l_ident=None, dsname=None):
        self.all_examples = []
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.holdout_examples = []
        self.after_examples = []
        self.before_examples = []

        self.all_batches = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.holdout_batches = []
        self.after_batches = []

        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.dsname = dsname
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.l_ident = 'targets'
        self.read_dataset(all_src)


    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()
        self.initialize_after_batch()
        self.initialize_before_batch()

    def load_splits(self, all_src, splits_src, mega_cool_mode, shport):
        splits_df = pd.read_csv(splits_src)

        self.after_examples = []
        self.after_batches = []
        self.before_examples = []
        self.before_batches = []
        # splits_df = splits_df.head(10000)
        not_contain_after_flag = 1
        # print("splits_df.columns:", splits_df.columns)
        if "index" in splits_df.columns:
            splits = splits_df.set_index("index")["split"].to_dict()
        elif "combined" in all_src:
            if mega_cool_mode:
                if "version" not in splits_df.columns:
                    print("adding dummy version")
                splits_df["version"] = "before"
                splits = splits_df.fillna("before").set_index(["dataset", "idx", "version"])["split"].to_dict()
            else:
                splits = splits_df.set_index(["dataset", "example_index"])["split"].to_dict()
        else:
            splits = splits_df.set_index("example_index")["split"].to_dict()
        split_examples = {
            "train": self.train_examples,
            "valid": self.valid_examples,
            "test": self.test_examples,
            "holdout": self.holdout_examples,
            "after": self.after_examples,
            "before": self.before_examples,
        }

        if "combined" in all_src:
            missing_src = splits_src + ".missing.txt"
            missing_keys = []
            after_src = splits_src + ".after.txt"
            after_keys = []

            new_splits = {}
            with jsonlines.open(all_src) as reader:
                if shport:
                    reader = itertools.islice(reader, 1000)
                print("load json -> msr mapping...")
                for json_idx, entry in enumerate(reader):
                    file_name = entry["file_name"]
                    version = "before"
                    if "_after_" in file_name:
                        version = "after"
                    if version == "after" and not mega_cool_mode:
                        after_keys.append((json_idx, file_name))
                        continue
                    dataset, real_file_name = file_name.split("/")
                    example_idx = int(real_file_name.split("_")[0])
                    if dataset == "Devign":
                        example_idx -= 1
                    if mega_cool_mode:
                        source_idx = (dataset, example_idx, version)
                    else:
                        source_idx = (dataset, example_idx)
                    try:
                        split_idx = splits[source_idx]
                    except KeyError:
                        missing_keys.append(source_idx)
                        continue
                    new_splits[json_idx] = split_idx
                print("done loading json -> msr mapping")
                if shport:
                    print(json.dumps(new_splits))
            splits = new_splits

            print(len(missing_keys), "keys missing from splits.csv")
            with open(missing_src, "w") as f:
                f.write("\n".join(str(m) for m in missing_keys))

            print(len(after_keys), "keys skipped because _after_")
            with open(after_src, "w") as f:
                f.write("\n".join(",".join([str(mm) for mm in m]) for m in after_keys))

        # if "MSR" in all_src:
        if True:
            missing_src = splits_src + ".missing.txt"
            missing_keys = []
            after_src = splits_src + ".after.txt"
            after_keys = []

            new_splits = {}
            new_splits_after = {}
            new_splits_after_file_name = {}
            new_splits_file_name = {}
            with jsonlines.open(all_src) as reader:
                print("load json -> msr mapping...")
                for json_idx, entry in enumerate(reader):
                    # if json_idx >= 1000:
                    #     break
                    file_name = entry["file_name"]
                    if not_contain_after_flag:
                        if "_after_" in file_name:
                            after_keys.append((json_idx, file_name))
                            # continue

                    source_idx = int(file_name.split("_")[0])
                    try:
                        split_idx = splits[source_idx]
                    except KeyError:
                        missing_keys.append(source_idx)
                        continue
                    if "_after_" in file_name:
                        new_splits_after[json_idx] = 'after'
                        new_splits_after_file_name[json_idx] = file_name
                    if "_before_" in file_name:
                        new_splits[json_idx] = split_idx
                        new_splits_file_name[json_idx] = file_name
                print("done loading json -> msr mapping")
            splits = new_splits

            print(len(missing_keys), "keys missing from splits.csv")
            with open(missing_src, "w") as f:
                f.write("\n".join(str(m) for m in missing_keys))

            print(len(after_keys), "keys skipped because _after_")
            with open(after_src, "w") as f:
                f.write("\n".join(",".join([str(mm) for mm in m]) for m in after_keys))

        # print("splits:", json.dumps(splits))
        splits_reveal = {}
        skipped_keys = 0
        printed_keys = 0
        for i, example in enumerate(self.all_examples):
            # print("i:", i)
            json_idx = example._id
            file_name = example.file_name
            # print("file_name:", file_name)
            if "before" in file_name:
                if json_idx in splits:
                    split_examples[splits[json_idx]].append(example)
                    splits_reveal[i] = splits[json_idx]
                split_examples['before'].append(example)
            if 'after' in file_name:
                # if json_idx in new_splits_after:
                split_examples['after'].append(example)

            # if printed_keys < 10:
            #     print("PRINTING DEBUG", printed_keys, json_idx, splits[json_idx], example._id, example.num_nodes, example.target, example.graph)
            #     printed_keys += 1
            # else:
            #     skipped_keys += 1
        # print("skipped", skipped_keys, "keys")
        self.initialize_dataset()
        # exit()
        return splits_reveal

    def read_dataset(self, all_src):

        # data = []
        #
        # # 读取 JSONL 文件并将每行数据添加到列表中
        # with jsonlines.open(all_src) as reader:
        #     for i, obj in enumerate(reader):
        #         print(i,"    ")
        #         data.append(obj)
        # print("22222")
        # df = pd.DataFrame(data)
        # # for item in df['file_name'].tolist():
        # #     print(item)
        # # 创建一个新列，其中去除 '_after_' 和 '_before_'
        # df['ModifiedColumn'] = df['file_name'].str.replace('_after_', '').str.replace('_before_', '')
        #
        # # 找出 'ModifiedColumn' 中的重复行
        # duplicates = df['ModifiedColumn'].duplicated(keep=False)
        #
        # # 筛选出包含重复项的行
        # filtered_df = df[duplicates]
        #
        # filtered_df = filtered_df.sort_values(by='ModifiedColumn')
        # print("111122222")
        # for i, entry in filtered_df.iterrows():
        #     if i >= 1000:
        #         break
        #
        #
        #     # if 'after' in entry['file_name']:
        #     #     after_files.append(entry['file_name'])
        #     example = DataEntry(
        #         _id=i,
        #         datset=self,
        #         num_nodes=len(entry[self.n_ident]),
        #         features=entry[self.n_ident],
        #         edges=entry[self.g_ident],
        #         target=entry[self.l_ident][0][0],
        #         file_name=entry['file_name']
        #     )
        #     self.all_examples.append(example)
        # # exit()



        debug('Reading all data File!')
        with jsonlines.open(all_src) as reader:
            for i, entry in enumerate(tqdm(reader, desc="read data")):
                example = DataEntry(
                    _id=i,
                    datset=self,
                    num_nodes=len(entry[self.n_ident]),
                    features=entry[self.n_ident],
                    edges=entry[self.g_ident],
                    target=entry[self.l_ident][0][0],
                    file_name=entry['file_name']
                )
                self.all_examples.append(example)


            # for file_name in after_files:
            #     file_name_1, file_name_2 = re.split("after", file_name)





    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)
        pass

    def initialize_after_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.after_batches = initialize_batch(self.after_examples, batch_size)
        return len(self.after_batches)
        pass

    def initialize_before_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.before_batches = initialize_batch(self.before_examples, batch_size)
        return len(self.before_batches)
        pass

    def initialize_holdout_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.holdout_batches = initialize_batch(self.holdout_examples, batch_size)
        return len(self.holdout_batches)
        pass

    def initialize_all_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.all_batches = initialize_batch(self.all_examples, batch_size)
        return len(self.all_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids, use_cache=False):
        unique_ids = ",".join(map(str, sorted(set(ids))))
        if use_cache and unique_ids in self.batch_cache:
            return self.batch_cache[unique_ids]
        else:
            taken_entries = [entries[i] for i in ids]
            labels = [e.target for e in taken_entries]
            batch_graph = dgl.batch([entry.graph for entry in taken_entries])
            return batch_graph, torch.FloatTensor(labels)

    def get_dataset_by_ids_for_GGNN_v2(self, entries, ids, use_cache=False):
        unique_ids = ",".join(map(str, sorted(set(ids))))
        if use_cache and unique_ids in self.batch_cache:
            return self.batch_cache[unique_ids]
        else:
            taken_entries = [entries[i] for i in ids]
            labels = [e.target for e in taken_entries]
            files_name =  [e.file_name for e in taken_entries]
            batch_graph = dgl.batch([entry.graph for entry in taken_entries])
            return batch_graph, torch.FloatTensor(labels), files_name


    def get_dataset_by_ids_for_GGNN_Synthetic(self, entries, ids, use_cache=False):
        unique_ids = ",".join(map(str, sorted(set(ids))))
        if use_cache and unique_ids in self.batch_cache:
            return self.batch_cache[unique_ids]
        else:
            taken_entries = [entries[i] for i in ids]
            labels = [e.target for e in taken_entries]
            files_name = [e.file_name for e in taken_entries]
            batch_graph = dgl.batch([entry.graph for entry in taken_entries])
            return batch_graph, torch.FloatTensor(labels), files_name

    def get_next_train_batch(self):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)

    def get_next_test_batch_v2(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN_v2(self.test_examples, ids)

    def get_next_after_batch_v2(self):
        if len(self.after_batches) == 0:
            self.initialize_afterbatch()
        ids = self.after_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN_v2(self.after_examples, ids)

    def get_next_before_batch_v2(self):
        if len(self.before_batches) == 0:
            self.initialize_before_batch()
        ids = self.before_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN_v2(self.before_examples, ids)

    def get_next_holdout_batch(self):
        if len(self.holdout_batches) == 0:
            self.initialize_test_batch()
        ids = self.holdout_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN(self.holdout_examples, ids)

    def get_next_all_batch(self):
        if len(self.all_batches) == 0:
            self.initialize_all_batch()
        ids = self.all_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN(self.all_examples, ids)

    def get_next_all_batch_synthetic(self):
        if len(self.all_batches) == 0:
            self.initialize_all_batch()
        ids = self.all_batches.pop(0)
        return self.get_dataset_by_ids_for_GGNN_Synthetic(self.all_examples, ids)

def test_ds():
    c_root = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/Synthetic_V3/full_experiment_real_data_processed/vSynthetic_V3'
    ds = DataSet(Path(c_root + "/train_GGNNinput.json"),
                 n_ident='node_features',
                 g_ident='graph',
                 l_ident='targets',
                 dsname="devign")
    assert ds.feature_size == ds.train_examples[0].ndata["features"].shape[1]
    assert ds.max_edge_type == max(g.edata["etype"].max().item() for g in (ds.train_examples + ds.valid_examples + ds.test_examples))


if __name__ == '__main__':
    test_ds()