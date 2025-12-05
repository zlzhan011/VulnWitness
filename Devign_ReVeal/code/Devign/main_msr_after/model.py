import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f
from data_loader.batch_graph import get_network_inputs, de_batchify_graphs


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3, padding=0)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph, cuda=False, save_after_ggnn=False):
        graph, features, edge_types = get_network_inputs(graph, cuda=cuda, device="cuda:0")
        outputs = self.ggnn(graph, features, edge_types)
        # print("outputs shape:", outputs.shape)
        # print("features shape:", features.shape)
        x_i = de_batchify_graphs(graph, features)
        h_i = de_batchify_graphs(graph, outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        # print("h_i shape:", h_i.shape)
        # print("c_i shape:", c_i.shape)
        # print("batch_size:", batch_size)

        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        if Y_1.shape[-1] < 2:
            Y_1 = f.pad(Y_1, (0, 2 - Y_1.shape[-1]), mode='constant', value=0)
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )

        if Z_1.shape[-1] < 2:
            Z_1 = f.pad(Z_1, (0, 2 - Z_1.shape[-1]), mode='constant', value=0)

        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result


    def forward_features(self, graph, cuda=False, save_after_ggnn=False):
        feature_list = []
        graph, features, edge_types = get_network_inputs(graph, cuda=cuda, device="cuda:0")
        outputs = self.ggnn(graph, features, edge_types)
        x_i = de_batchify_graphs(graph, features)
        h_i = de_batchify_graphs(graph, outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        feature_list.append(before_avg)
        avg = before_avg.mean(dim=1)
        feature_list.append(avg)
        result = self.sigmoid(avg).squeeze(dim=-1)
        return result, feature_list



class DevignModelMultiCategory(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(DevignModelMultiCategory, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=7)
        self.mlp_y = nn.Linear(in_features=self.out_dim, out_features=7)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, graph, cuda=False, save_after_ggnn=False, save_mid_result=False):
        graph, features, edge_types = get_network_inputs(graph, cuda=cuda, device="cuda:0")
        outputs = self.ggnn(graph, features, edge_types)
        features_list = []
        x_i = de_batchify_graphs(graph, features)
        h_i = de_batchify_graphs(graph, outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        features_list.append(before_avg)
        avg = before_avg.mean(dim=1)
        features_list.append(avg)
        # result = self.sigmoid(avg).squeeze(dim=-1)
        result = nn.functional.softmax(avg, dim=1)
        features_list.append(result)
        if save_mid_result:
            return result, features_list
        else:
            return result


    def forward_features(self, graph, cuda=False, save_after_ggnn=False):
        feature_list = []
        graph, features, edge_types = get_network_inputs(graph, cuda=cuda, device="cuda:0")
        outputs = self.ggnn(graph, features, edge_types)
        x_i = de_batchify_graphs(graph, features)
        h_i = de_batchify_graphs(graph, outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        feature_list.append(before_avg)
        avg = before_avg.mean(dim=1)
        feature_list.append(avg)

        # result = self.sigmoid(avg).squeeze(dim=-1)
        result = nn.functional.softmax(avg, dim=1)
        feature_list.append(result)

        return result, feature_list

class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph, cuda=False, save_after_ggnn=False):
        graph, features, edge_types = get_network_inputs(graph, cuda=cuda, device="cuda:0")
        outputs = self.ggnn(graph, features, edge_types)
        h_i = de_batchify_graphs(graph, outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        if save_after_ggnn:
            features = h_i.mean(dim=1)
            assert features.shape == (h_i.shape[0], h_i.shape[2])
            return result, features
        else:
            return result

    def forward_features(self, graph, cuda=False, save_after_ggnn=False):
        feature_list = []
        graph, features, edge_types = get_network_inputs(graph, cuda=cuda, device="cuda:0")
        outputs = self.ggnn(graph, features, edge_types)
        h_i = de_batchify_graphs(graph, outputs)
        # print("h_i.shape:", h_i.shape)
        feature_list.append(h_i)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        # print("ggnn_sum.shape:", ggnn_sum.shape)
        feature_list.append(ggnn_sum)
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        if save_after_ggnn:
            features = h_i.mean(dim=1)
            assert features.shape == (h_i.shape[0], h_i.shape[2])
            return result, features
        else:
            return result, feature_list



class GGNNSumMulCategory(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSumMulCategory, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=7)
        # self.classifier = nn.Linear(in_features=output_dim, out_features=num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, graph, cuda=False, save_after_ggnn=False, save_mid_result=False):
        graph, features, edge_types = get_network_inputs(graph, cuda=cuda, device="cuda:0")
        features_list =[]
        outputs = self.ggnn(graph, features, edge_types)
        h_i = de_batchify_graphs(graph, outputs)
        features_list.append(h_i)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        features_list.append(ggnn_sum)
        # result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        result = nn.functional.softmax(ggnn_sum, dim=1)
        features_list.append(result)
        if save_after_ggnn:
            features = h_i.mean(dim=1)
            assert features.shape == (h_i.shape[0], h_i.shape[2])
            return result, features
        else:
            # return result
            if save_mid_result:
                return result, features_list
            else:
                return result

    def forward_features(self, graph, cuda=False, save_after_ggnn=False):
        feature_list = []
        graph, features, edge_types = get_network_inputs(graph, cuda=cuda, device="cuda:0")
        outputs = self.ggnn(graph, features, edge_types)
        h_i = de_batchify_graphs(graph, outputs)
        # print("h_i.shape:", h_i.shape)
        feature_list.append(h_i)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        # print("ggnn_sum.shape:", ggnn_sum.shape)
        feature_list.append(ggnn_sum)
        # result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        result = nn.functional.softmax(ggnn_sum, dim=1)
        feature_list.append(result)
        if save_after_ggnn:
            features = h_i.mean(dim=1)
            assert features.shape == (h_i.shape[0], h_i.shape[2])
            return result, features
        else:
            return result, feature_list
