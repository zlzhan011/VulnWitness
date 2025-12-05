# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob


import torch
import torch.nn as nn


# class MultiCategoryModel(nn.Module):
#     def __init__(self, encoder, config, tokenizer, args, num_classes):
#         super(MultiCategoryModel, self).__init__()
#         self.encoder = encoder
#         self.config = config
#         self.tokenizer = tokenizer
#         self.args = args
#         self.num_classes = num_classes
#
#         # 修改输出层以适应多分类
#         self.classifier = nn.Linear(self.config.hidden_size, self.num_classes)
#
#     def forward(self, input_ids=None, labels=None):
#         outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
#
#         # 应用分类器
#         logits = self.classifier(outputs[:, 0, :])  # 假设使用第一个token（如CLS）进行分类
#
#         # 多分类问题通常使用softmax
#         prob = torch.softmax(logits, dim=1)
#
#         if labels is not None:
#             # 使用交叉熵损失
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
#             return loss, prob
#         else:
#             return prob



class MultiCategoryModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args, num_classes):
        super(MultiCategoryModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.num_classes = num_classes
        self.config.hidden_size = 32
        self.classifier = nn.Linear(self.config.hidden_size, self.num_classes)


    def forward(self,input_ids=None, labels=None, input_embed=None,  output_attentions=False, output_features_list=False):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1),
                                               output_attentions=output_attentions)
            else:
                outputs = self.encoder(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            print("last_hidden_state:", last_hidden_state.shape)
            # logits = self.classifier(last_hidden_state)
            logits = last_hidden_state
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            features_list = []
            if input_ids is not None:
                outputs = \
                self.encoder(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            # logits = self.classifier(outputs)

            logits = outputs
            features_list.append(logits)
            prob = torch.softmax(logits, dim=-1)
            features_list.append(prob)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                if output_features_list:
                    return loss, prob, features_list
                else:
                    return loss, prob
            else:
                if output_features_list:
                    return prob, features_list
                else:
                    return prob

    def feature_list(self, input_ids=None, labels=None, input_embed=None,  output_attentions=False):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1),
                                               output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            features_list = []
            output_attentions = True
            output_hidden_states=True
            # para = []
            # if input_ids is not None:
            #     outputs_original = \
            #     self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions, output_hidden_states=output_hidden_states)
            #     outputs = outputs_original[0]
            #     hidden_states = outputs_original.hidden_states
            #     attentions = outputs_original.attentions
            # else:
            #     outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]


            if input_ids is not None:
                outputs_original = \
                self.encoder(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
                outputs = outputs_original[0]
                hidden_states = outputs
                # attentions = outputs_original.attentions
            else:
                outputs = self.encoder(inputs_embeds=input_embed, output_attentions=output_attentions)[0]


            # para.append(outputs)
            # logits = self.classifier(outputs)
            # para.append(logits)
            # logits = self.classifier(outputs)
            print(outputs.shape)
            print(self.config.hidden_size, self.num_classes)
            logits = self.classifier(outputs)

            # hidden_states = [item.mean(dim=1) for item in hidden_states]
            features_list.append(hidden_states)
            features_list.append(logits)
            prob = torch.softmax(logits, dim=-1)
            features_list.append(prob)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, features_list
            else:
                return logits, prob, features_list

        
 
