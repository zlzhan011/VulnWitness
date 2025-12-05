import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import copy
# Be_Ue_final_metrics = {'CodeBERT': {'original': {'tn': 17447, 'fp': 362, 'fn': 727, 'tp': 328, 'recall': 0.3109004739336493, 'precision': 0.4753623188405797, 'f1': 0.37593123209169055}, 'add_after_to_before': {'tn': 17553, 'fp': 256, 'fn': 787, 'tp': 268, 'recall': 0.25402843601895736, 'precision': 0.5114503816793893, 'f1': 0.3394553514882837}}, 'LineVul': {'original': {'tn': 17724, 'fp': 85, 'fn': 212, 'tp': 843, 'recall': 0.7990521327014218, 'precision': 0.9084051724137931, 'f1': 0.8502269288956127}, 'add_after_to_before': {'tn': 17785, 'fp': 24, 'fn': 388, 'tp': 667, 'recall': 0.6322274881516587, 'precision': 0.9652677279305355, 'f1': 0.7640320733104238}}, 'DeepDFA_Plus_LineVul': {'original': {'tn': 17768, 'fp': 40, 'fn': 65, 'tp': 841, 'recall': 0.9282560706401766, 'precision': 0.9545970488081725, 'f1': 0.9412423055400111}, 'add_after_to_before': {'tn': 17788, 'fp': 20, 'fn': 225, 'tp': 681, 'recall': 0.7516556291390728, 'precision': 0.9714693295292439, 'f1': 0.8475420037336652}}, 'DeepDFA_Only': {'original': {'tn': 17694, 'fp': 114, 'fn': 422, 'tp': 484, 'recall': 0.5342163355408388, 'precision': 0.8093645484949833, 'f1': 0.6436170212765957}, 'add_after_to_before': {'tn': 17694, 'fp': 114, 'fn': 422, 'tp': 484, 'recall': 0.5342163355408388, 'precision': 0.8093645484949833, 'f1': 0.6436170212765957}}, 'LineVul_Process_func': {'original': {'tn': 17760, 'fp': 49, 'fn': 130, 'tp': 925, 'recall': 0.8767772511848341, 'precision': 0.9496919917864476, 'f1': 0.9117792015771315}, 'add_after_to_before': {'tn': 17678, 'fp': 77, 'fn': 118, 'tp': 991, 'recall': 0.8935978358881875, 'precision': 0.9279026217228464, 'f1': 0.9104271933853927}}, 'ReVeal': {'original': {'tn': 16556, 'fp': 472, 'fn': 729, 'tp': 258, 'recall': 0.2613981762917933, 'precision': 0.35342465753424657, 'f1': 0.30052417006406523}, 'add_after_to_before': {'tn': 16713, 'fp': 315, 'fn': 771, 'tp': 216, 'recall': 0.2188449848024316, 'precision': 0.4067796610169492, 'f1': 0.2845849802371541}}, 'Devign': {'original': {'tn': 16346, 'fp': 682, 'fn': 729, 'tp': 258, 'recall': 0.2613981762917933, 'precision': 0.274468085106383, 'f1': 0.2677737415672029}, 'add_after_to_before': {'tn': 17389, 'fp': 602, 'fn': 879, 'tp': 108, 'recall': 0.1094224924012158, 'precision': 0.15211267605633802, 'f1': 0.12728344136711844}}}
Be_Ae_final_metrics = {'CodeBERT': {'original': {'tn': 764, 'fp': 291, 'fn': 727, 'tp': 328, 'recall': 0.3109004739336493, 'precision': 0.529886914378029, 'f1': 0.39187574671445635}, 'add_after_to_before': {'tn': 937, 'fp': 118, 'fn': 787, 'tp': 268, 'recall': 0.25402843601895736, 'precision': 0.694300518134715, 'f1': 0.3719639139486468}},
                       'LineVul': {'original': {'tn': 236, 'fp': 819, 'fn': 212, 'tp': 843, 'recall': 0.7990521327014218, 'precision': 0.5072202166064982, 'f1': 0.6205373573794626}, 'add_after_to_before': {'tn': 635, 'fp': 420, 'fn': 388, 'tp': 667, 'recall': 0.6322274881516587, 'precision': 0.6136154553817847, 'f1': 0.6227824463118581}},
                       'DeepDFA_Plus_LineVul': {'original': {'tn': 67, 'fp': 839, 'fn': 65, 'tp': 841, 'recall': 0.9282560706401766, 'precision': 0.5005952380952381, 'f1': 0.6504253673627224}, 'add_after_to_before': {'tn': 485, 'fp': 421, 'fn': 225, 'tp': 681, 'recall': 0.7516556291390728, 'precision': 0.6179673321234119, 'f1': 0.6782868525896413}},
                       'DeepDFA_Only': {'original': {'tn': 513, 'fp': 393, 'fn': 422, 'tp': 484, 'recall': 0.5342163355408388, 'precision': 0.5518814139110604, 'f1': 0.5429052159282107}, 'add_after_to_before': {'tn': 513, 'fp': 393, 'fn': 422, 'tp': 484, 'recall': 0.5342163355408388, 'precision': 0.5518814139110604, 'f1': 0.5429052159282107}},
                       'LineVul_Process_func': {'original': {'tn': 354, 'fp': 701, 'fn': 130, 'tp': 925, 'recall': 0.8767772511848341, 'precision': 0.568880688806888, 'f1': 0.6900410294666169}, 'add_after_to_before': {'tn': 315, 'fp': 794, 'fn': 118, 'tp': 991, 'recall': 0.8935978358881875, 'precision': 0.5551820728291317, 'f1': 0.6848652384243262}},
                       'ReVeal': {'original': {'tn': 696, 'fp': 213, 'fn': 662, 'tp': 247, 'recall': 0.2717271727172717, 'precision': 0.5369565217391304, 'f1': 0.36084733382030676}, 'add_after_to_before': {'tn': 770, 'fp': 139, 'fn': 708, 'tp': 201, 'recall': 0.22112211221122113, 'precision': 0.5911764705882353, 'f1': 0.32185748598879105}},
                       'Devign': {'original': {'tn': 715, 'fp': 194, 'fn': 669, 'tp': 240, 'recall': 0.264026402640264, 'precision': 0.5529953917050692, 'f1': 0.3574087862993299}, 'add_after_to_before': {'tn': 884, 'fp': 25, 'fn': 820, 'tp': 89, 'recall': 0.09790979097909791, 'precision': 0.7807017543859649, 'f1': 0.17399804496578689}},
                       'LLama3': {'original': {'tn': 42, 'fp': 1005, 'fn': 33, 'tp': 1014, 'recall': 0.9684813753581661, 'precision': 0.5022288261515602, 'f1': 0.6614481409001957}, 'add_after_to_before': {'tn': 717, 'fp': 330, 'fn': 220, 'tp': 827, 'recall': 0.789875835721108, 'precision': 0.7147796024200519, 'f1': 0.7504537205081669}}}

paired_metrics = {"LLama3":{"precision":0.6923,"recall":0.7822,"f1":0.7345},
"LineVul":{"precision":0.6773,"recall":0.7422,"f1":0.6773},
"DeepDFA_Only":{"precision":0.5053,"recall":0.9956,"f1":0.6704},
"DeepDFA_Plus_LineVul":{"precision":0.5701,"recall":0.9647,"f1":0.7167},
"ReVeal":{"precision":0.5316,"recall":0.9119,"f1":0.6716},
"Devign":{"precision":0.5214,"recall":0.9747,"f1":0.6794},
"CodeBERT":{"precision":0.6096,"recall":0.7701,"f1":0.6805},
"LineVul_Process_func":{"precision":0.9303,"recall":0.8976,"f1":0.9137}
          }

for k, v in Be_Ae_final_metrics.items():
    paired_v = paired_metrics[k]
    Be_Ae_final_metrics[k]['train_on_paired'] = paired_v




metrics_names = ['precision', 'recall', 'f1']
models_name = [ 'DeepDFA_Plus_LineVul', 'DeepDFA_Only', 'ReVeal',
               'Devign','CodeBERT','LLama3','LineVul', 'LineVul_Process_func']
# models_name = ['Devign']
virsons_name = ['original', 'add_after_to_before', 'train_on_paired']  # original  add_after_to_before
for m_result in [Be_Ae_final_metrics]:
  print("m_result:", m_result)
  for one_metrics_name in metrics_names:
      one_metrics_all_model_result = []
      for model_name in models_name:
          one_model_result = []
          for version_name in virsons_name:
              one_metrics_value = m_result[model_name][version_name][one_metrics_name]
              one_model_result.append(one_metrics_value)
          one_metrics_all_model_result.append(one_model_result)

      data = one_metrics_all_model_result
      # 转换数据结构，以适应绘图需求
      data = np.array(data)

      # 设置直方图的位置偏移量，使得每对数据靠近，而不同对数据之间有间隔
      # x = np.arange(len(data)) * 2  # 每对数据之间的基础间隔
      # width = 0.60  # 每个直方图的宽度

      x = np.arange(len(data)) * 2.5  # 增加间隔以适应三个条形图
      width = 0.7 / 3  # 调整宽度，使三个条形图能够并排显示

      fig, ax = plt.subplots()
      rects1 = ax.bar(x - width, data[:, 0], width, label=virsons_name[0])
      rects2 = ax.bar(x, data[:, 1], width, label=virsons_name[1])
      rects3 = ax.bar(x + width, data[:, 2], width, label=virsons_name[2])
      # 添加一些文本标签，标题和自定义x轴刻度标签等
      ax.set_xlabel('models name')
      ax.set_ylabel('scores')
      ax.set_title(one_metrics_name)
      ax.set_xticks(x)
      models_name_xtick = copy.deepcopy(models_name)
      DeepDFA_Plus_LineVul_index = models_name_xtick.index('DeepDFA_Plus_LineVul')
      models_name_xtick[DeepDFA_Plus_LineVul_index] = 'DeepDFA_LineVul'
      ax.set_xticklabels(models_name_xtick, rotation=10)
      ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.0))


      # 设置y轴的刻度为0.05的倍数
      ax.yaxis.set_major_locator(MultipleLocator(0.05))

      # 为每一个y轴的刻度添加水平参考线
      y_ticks = np.arange(0, 1.05, 0.05)  # 从0到1（含）以0.05为步长
      for tick in y_ticks:
          ax.axhline(y=tick, color='gray', linewidth=0.5, linestyle='--')

      plt.show()