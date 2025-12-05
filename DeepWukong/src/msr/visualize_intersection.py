import os.path
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def expand_ranges(ranges):
    """Expand list of {'start_offset': int, 'end_offset': int} to a set of indices."""
    indices = set()
    for r in ranges:
        indices.update(range(r['start_offset'], r['end_offset']))
    return indices

def draw_code_with_underlines_scaled(text, red_ranges, black_ranges):
    red_indices = expand_ranges(red_ranges)
    black_indices = expand_ranges(black_ranges)

    fig, ax = plt.subplots(figsize=(16, 20))  # Increased figure size
    ax.axis('off')

    # Further reduced sizes for tight layout
    line_spacing = 0.03
    char_spacing = 0.01
    char_width = 0.01
    font_size = 12
    underline_offset = line_spacing * 0.3  # 红线在下
    doubleline_gap = line_spacing * 0.2  # 黑线再下方一点

    lines = text.split('\n')
    y = 1.0
    char_index = 0

    for line in lines:
        x = 0.0
        for char in line:
            ax.text(x, y, char, fontsize=font_size, va='center', family='monospace')

            if char_index in red_indices:
                ax.plot([x, x + char_width], [y - underline_offset, y - underline_offset], color='red', linewidth=1)

            if char_index in black_indices:
                offset = -underline_offset - doubleline_gap if char_index in red_indices else -underline_offset
                ax.plot([x, x + char_width], [y + offset, y + offset], color='black', linewidth=1)

            x += char_spacing
            char_index += 1

        char_index += 1  # account for \n
        y -= line_spacing

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


# # 示例数据
# text = "Hel\nlo,\nworld!"
#
# red_ranges = [{"start_index": 1, "end_index": 2}, {"start_index": 5, "end_index": 5}]
# black_ranges = [{"start_index": 0, "end_index": 0}, {"start_index": 10, "end_index": 12}]
#
# draw_colored_underline_multiline(text, red_ranges, black_ranges)




def check_res():
    file = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_Intersection_Visual_LineVul_Map/186678/array/6.xfg.pkl.linevul_map.json.pkl'
    file = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_Intersection_Visual_LineVul_Map/184827/array/10.xfg.pkl.linevul_map.json.pkl'
    with open(file, 'rb') as f:
        content = pickle.load(f)
        text = content['text']
        red_ranges = content['red_ranges']
        black_ranges = content['black_ranges']
        print("text:\n", text)
        print("\n\nred_ranges:\n", red_ranges)
        for item in red_ranges:
            print(item)
        print("\n\nblack_ranges:\n", )
        for item in black_ranges:
            print(item)


if __name__ == '__main__':

    check_res()
    exit()


    c_file_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/source-code/test/chrome/186678/186678_func_before_target_1.c'
    with open(c_file_path, "r") as f:
        original_code = f.read()



    import ast
    import json
    xfg_path = '/scratch/c00590656/vulnerability/DeepWukong/data/msr/XFG_LineVul_Map/186678/array/6.xfg.pkl.linevul_map.json'
    with open(xfg_path, 'r') as f:
        content = f.read()
    content = content.replace("nan", "None")
    # 使用 ast.literal_eval 尝试安全地解析成 Python 字典
    one_slicing_res = ast.literal_eval(content)
    print(one_slicing_res)
    node_mapped_linevul_tokens_concat = []
    for node_index, node_corr_information in one_slicing_res.items():
        node_mapped_linevul_tokens = node_corr_information['node_mapped_linevul_tokens']
        node_mapped_linevul_tokens_concat += node_mapped_linevul_tokens



    linevul_token_scores_dir = '/scratch/c00590656/vulnerability/LineVul/linevul/shap/saved_models_MSR/token_weight_map_predict_positive_label_positive'
    lineVul_res_path = os.path.join(linevul_token_scores_dir,'186678.pkl.xlsx' )
    # lineVul_res = [{'token': nan, 'weight': 0.0001207038294523954}, {'token': 'int', 'weight': -0.005734428996220231}, {'token': 'PDF', 'weight': 0.001018951879814267}, {'token': 'ium', 'weight': 0.001645873533561826}, {'token': 'Engine', 'weight': -0.001255324492896242}, {'token': '::', 'weight': -0.003272402017111225}, {'token': 'Get', 'weight': 0.001497774933730918}, {'token': 'V', 'weight': -3.765701382820071e-05}, {'token': 'isible', 'weight': -0.001232636582461141}, {'token': 'Page', 'weight': -0.000541349362936757}, {'token': 'Index', 'weight': -0.004283890549448274}, {'token': '(', 'weight': 0.01246792916208506}, {'token': 'F', 'weight': 0.003192711155861616}, {'token': 'PDF', 'weight': 0.003192711155861616}, {'token': '_', 'weight': 0.0028838817961514}, {'token': 'PA', 'weight': 0.0028838817961514}, {'token': 'GE', 'weight': 0.008265097004671892}, {'token': 'page', 'weight': 0.006096220885713895}, {'token': ')', 'weight': 0.008029504989584288}, {'token': '{', 'weight': -0.001604273449629545}, {'token': '\n', 'weight': -0.001910895574837923}, {'token': nan, 'weight': -0.001279089134186506}, {'token': 'for', 'weight': -0.001376800704747438}, {'token': '(', 'weight': -0.003850359003990889}, {'token': 'int', 'weight': -0.007432662416249514}, {'token': 'page', 'weight': 0.007654893212020397}, {'token': '_', 'weight': 0.01025454793125391}, {'token': 'index', 'weight': -0.003516454914850848}, {'token': ':', 'weight': -0.001022070380193847}, {'token': 'visible', 'weight': 0.003033321151243789}, {'token': '_', 'weight': 0.000680880421506507}, {'token': 'pages', 'weight': 0.003746465269830965}, {'token': '_', 'weight': 0.001179429192450785}, {'token': ')', 'weight': 0.009639263108727477}, {'token': '{', 'weight': -0.001678298693150282}, {'token': '\n', 'weight': -0.001218170393258333}, {'token': nan, 'weight': 0.006890990305691957}, {'token': nan, 'weight': 0.004747051279991865}, {'token': nan, 'weight': 0.005059553775936365}, {'token': nan, 'weight': 0.007985693868249655}, {'token': 'if', 'weight': 0.002922867890447378}, {'token': '(', 'weight': 0.002770951483398676}, {'token': 'pages', 'weight': 0.0003524263322885548}, {'token': '_', 'weight': -0.001072414319163987}, {'token': '[', 'weight': -0.0007515145532254661}, {'token': 'page', 'weight': -0.0007515145532254661}, {'token': '_', 'weight': -0.0009297403573457683}, {'token': 'index', 'weight': -0.0009297403573457683}, {'token': ']', 'weight': -0.0007608654975358928}, {'token': '->', 'weight': -0.0007608654975358928}, {'token': 'Get', 'weight': 0.002558043514866205}, {'token': 'Page', 'weight': 0.002558043514866205}, {'token': '()', 'weight': 0.001563989845592351}, {'token': nan, 'weight': 0.001563989845592351}, {'token': 'page', 'weight': 0.004372302731055589}, {'token': ')', 'weight': 0.002573523027378889}, {'token': '\n', 'weight': -0.002635042863683059}, {'token': nan, 'weight': -0.002635042863683059}, {'token': nan, 'weight': -0.00306125798334296}, {'token': nan, 'weight': -0.00306125798334296}, {'token': nan, 'weight': -0.003482058626384689}, {'token': nan, 'weight': -0.002497536112339451}, {'token': nan, 'weight': -0.005322838417039468}, {'token': 'return', 'weight': -0.004040248441294982}, {'token': 'page', 'weight': 0.002499426267324732}, {'token': '_', 'weight': 0.002499426267324732}, {'token': 'index', 'weight': 0.004425761773465917}, {'token': ';', 'weight': 0.004016799404500769}, {'token': '\n', 'weight': 0.00625060500505452}, {'token': nan, 'weight': 0.00625060500505452}, {'token': nan, 'weight': 0.008033039764716074}, {'token': '}', 'weight': 0.006258600642188237}, {'token': '\n', 'weight': -0.001769257179246499}, {'token': nan, 'weight': -0.0008674612555366295}, {'token': 'return', 'weight': -0.004056363302068069}, {'token': '-', 'weight': -0.003237610618368937}, {'token': '1', 'weight': 0.0005591937675116917}, {'token': ';', 'weight': 0.0005591937675116917}, {'token': '\n', 'weight': -0.002472581587827359}, {'token': '}', 'weight': -0.002472581587827359}, {'token': '\n', 'weight': 0.003798529259764995}, {'token': nan, 'weight': 0.001362387776279297}]
    lineVul_res = pd.read_excel(lineVul_res_path)

    lineVul_res_list = []
    for index, row in lineVul_res.iterrows():
        token = row['token']
        weight = row['weight']
        lineVul_res_list.append({'token': token, 'weight': weight, "start_offset":row['start_offset'],
                                 "end_offset":row['end_offset']})


    # one_slicing_res = {6: {'command': 'ANR', 'key': '11462', 'type': 'Statement', 'code': 'int', 'location': '2:7:64:66', 'functionId': '11457', 'childNum': '2', 'isCFGNode': 'True', 'operator': '', 'baseType': '', 'completeType': '', 'identifier': '', 'location_updated': '1#3', 'code_updated': 'int', 'node_mapped_linevul_tokens': [], 'node_average_weight': 0, 'node_sum_weight': 0}, 7: {'command': 'ANR', 'key': '11463', 'type': 'Label', 'code': 'page_index :', 'location': '2:11:68:79', 'functionId': '11457', 'childNum': '3', 'isCFGNode': 'True', 'operator': '', 'baseType': '', 'completeType': '', 'identifier': '', 'location_updated': '68#79', 'code_updated': 'page_index :', 'node_mapped_linevul_tokens': [{'token': 'page', 'start_offset': 68, 'end_offset': 72, 'weight': 0.007654893212020397}, {'token': '_', 'start_offset': 72, 'end_offset': 73, 'weight': 0.01025454793125391}, {'token': 'index', 'start_offset': 73, 'end_offset': 78, 'weight': -0.003516454914850848}], 'node_average_weight': 0.004797662076141153, 'node_sum_weight': 0.014392986228423459}, 9: {'command': 'ANR', 'key': '11465', 'type': 'Statement', 'code': 'visible_pages_', 'location': '2:24:81:94', 'functionId': '11457', 'childNum': '4', 'isCFGNode': 'True', 'operator': '', 'baseType': '', 'completeType': '', 'identifier': '', 'location_updated': '81#94', 'code_updated': 'visible_pages_', 'node_mapped_linevul_tokens': [{'token': 'visible', 'start_offset': 81, 'end_offset': 88, 'weight': 0.003033321151243789}, {'token': '_', 'start_offset': 88, 'end_offset': 89, 'weight': 0.000680880421506507}, {'token': 'pages', 'start_offset': 89, 'end_offset': 94, 'weight': 0.003746465269830965}], 'node_average_weight': 0.002486888947527087, 'node_sum_weight': 0.007460666842581261}, 1: {'command': 'ANR', 'key': '11457', 'type': 'Function', 'code': 'PDFiumEngine :: GetVisiblePageIndex', 'location': '1:1:1:191', 'functionId': '', 'childNum': '', 'isCFGNode': '', 'operator': '', 'baseType': '', 'completeType': '', 'identifier': '', 'location_updated': '0#193', 'code_updated': ' int PDFiumEngine::GetVisiblePageIndex(FPDF_PAGE page) {\n  for (int page_index : visible_pages_) {\n     if (pages_[page_index]->GetPage() == page)\n       return page_index;\n   }\n  return -1;\n}\n', 'node_mapped_linevul_tokens': [{'token': None, 'start_offset': 0, 'end_offset': 0, 'weight': 0.0001207038294523954}, {'token': None, 'start_offset': 0, 'end_offset': 0, 'weight': 0.001362387776279297}, {'token': 'int', 'start_offset': 1, 'end_offset': 4, 'weight': -0.005734428996220231}, {'token': 'PDF', 'start_offset': 5, 'end_offset': 8, 'weight': 0.001018951879814267}, {'token': 'ium', 'start_offset': 8, 'end_offset': 11, 'weight': 0.001645873533561826}, {'token': 'Engine', 'start_offset': 11, 'end_offset': 17, 'weight': -0.001255324492896242}, {'token': '::', 'start_offset': 17, 'end_offset': 19, 'weight': -0.003272402017111225}, {'token': 'Get', 'start_offset': 19, 'end_offset': 22, 'weight': 0.001497774933730918}, {'token': 'V', 'start_offset': 22, 'end_offset': 23, 'weight': -3.765701382820071e-05}, {'token': 'isible', 'start_offset': 23, 'end_offset': 29, 'weight': -0.001232636582461141}, {'token': 'Page', 'start_offset': 29, 'end_offset': 33, 'weight': -0.000541349362936757}, {'token': 'Index', 'start_offset': 33, 'end_offset': 38, 'weight': -0.004283890549448274}, {'token': '(', 'start_offset': 38, 'end_offset': 39, 'weight': 0.01246792916208506}, {'token': 'F', 'start_offset': 39, 'end_offset': 40, 'weight': 0.003192711155861616}, {'token': 'PDF', 'start_offset': 40, 'end_offset': 43, 'weight': 0.003192711155861616}, {'token': '_', 'start_offset': 43, 'end_offset': 44, 'weight': 0.0028838817961514}, {'token': 'PA', 'start_offset': 44, 'end_offset': 46, 'weight': 0.0028838817961514}, {'token': 'GE', 'start_offset': 46, 'end_offset': 48, 'weight': 0.008265097004671892}, {'token': 'page', 'start_offset': 49, 'end_offset': 53, 'weight': 0.006096220885713895}, {'token': ')', 'start_offset': 53, 'end_offset': 54, 'weight': 0.008029504989584288}, {'token': '{', 'start_offset': 55, 'end_offset': 56, 'weight': -0.001604273449629545}, {'token': '\n', 'start_offset': 56, 'end_offset': 57, 'weight': -0.001910895574837923}, {'token': None, 'start_offset': 58, 'end_offset': 58, 'weight': -0.001279089134186506}, {'token': 'for', 'start_offset': 59, 'end_offset': 62, 'weight': -0.001376800704747438}, {'token': '(', 'start_offset': 63, 'end_offset': 64, 'weight': -0.003850359003990889}, {'token': 'int', 'start_offset': 64, 'end_offset': 67, 'weight': -0.007432662416249514}, {'token': 'page', 'start_offset': 68, 'end_offset': 72, 'weight': 0.007654893212020397}, {'token': '_', 'start_offset': 72, 'end_offset': 73, 'weight': 0.01025454793125391}, {'token': 'index', 'start_offset': 73, 'end_offset': 78, 'weight': -0.003516454914850848}, {'token': ':', 'start_offset': 79, 'end_offset': 80, 'weight': -0.001022070380193847}, {'token': 'visible', 'start_offset': 81, 'end_offset': 88, 'weight': 0.003033321151243789}, {'token': '_', 'start_offset': 88, 'end_offset': 89, 'weight': 0.000680880421506507}, {'token': 'pages', 'start_offset': 89, 'end_offset': 94, 'weight': 0.003746465269830965}, {'token': '_', 'start_offset': 94, 'end_offset': 95, 'weight': 0.001179429192450785}, {'token': ')', 'start_offset': 95, 'end_offset': 96, 'weight': 0.009639263108727477}, {'token': '{', 'start_offset': 97, 'end_offset': 98, 'weight': -0.001678298693150282}, {'token': '\n', 'start_offset': 98, 'end_offset': 99, 'weight': -0.001218170393258333}, {'token': None, 'start_offset': 100, 'end_offset': 100, 'weight': 0.006890990305691957}, {'token': None, 'start_offset': 101, 'end_offset': 101, 'weight': 0.004747051279991865}, {'token': None, 'start_offset': 102, 'end_offset': 102, 'weight': 0.005059553775936365}, {'token': None, 'start_offset': 103, 'end_offset': 103, 'weight': 0.007985693868249655}, {'token': 'if', 'start_offset': 104, 'end_offset': 106, 'weight': 0.002922867890447378}, {'token': '(', 'start_offset': 107, 'end_offset': 108, 'weight': 0.002770951483398676}, {'token': 'pages', 'start_offset': 108, 'end_offset': 113, 'weight': 0.0003524263322885548}, {'token': '_', 'start_offset': 113, 'end_offset': 114, 'weight': -0.001072414319163987}, {'token': '[', 'start_offset': 114, 'end_offset': 115, 'weight': -0.0007515145532254661}, {'token': 'page', 'start_offset': 115, 'end_offset': 119, 'weight': -0.0007515145532254661}, {'token': '_', 'start_offset': 119, 'end_offset': 120, 'weight': -0.0009297403573457683}, {'token': 'index', 'start_offset': 120, 'end_offset': 125, 'weight': -0.0009297403573457683}, {'token': ']', 'start_offset': 125, 'end_offset': 126, 'weight': -0.0007608654975358928}, {'token': '->', 'start_offset': 126, 'end_offset': 128, 'weight': -0.0007608654975358928}, {'token': 'Get', 'start_offset': 128, 'end_offset': 131, 'weight': 0.002558043514866205}, {'token': 'Page', 'start_offset': 131, 'end_offset': 135, 'weight': 0.002558043514866205}, {'token': '()', 'start_offset': 135, 'end_offset': 137, 'weight': 0.001563989845592351}, {'token': None, 'start_offset': 138, 'end_offset': 140, 'weight': 0.001563989845592351}, {'token': 'page', 'start_offset': 141, 'end_offset': 145, 'weight': 0.004372302731055589}, {'token': ')', 'start_offset': 145, 'end_offset': 146, 'weight': 0.002573523027378889}, {'token': '\n', 'start_offset': 146, 'end_offset': 147, 'weight': -0.002635042863683059}, {'token': None, 'start_offset': 148, 'end_offset': 148, 'weight': -0.002635042863683059}, {'token': None, 'start_offset': 149, 'end_offset': 149, 'weight': -0.00306125798334296}, {'token': None, 'start_offset': 150, 'end_offset': 150, 'weight': -0.00306125798334296}, {'token': None, 'start_offset': 151, 'end_offset': 151, 'weight': -0.003482058626384689}, {'token': None, 'start_offset': 152, 'end_offset': 152, 'weight': -0.002497536112339451}, {'token': None, 'start_offset': 153, 'end_offset': 153, 'weight': -0.005322838417039468}, {'token': 'return', 'start_offset': 154, 'end_offset': 160, 'weight': -0.004040248441294982}, {'token': 'page', 'start_offset': 161, 'end_offset': 165, 'weight': 0.002499426267324732}, {'token': '_', 'start_offset': 165, 'end_offset': 166, 'weight': 0.002499426267324732}, {'token': 'index', 'start_offset': 166, 'end_offset': 171, 'weight': 0.004425761773465917}, {'token': ';', 'start_offset': 171, 'end_offset': 172, 'weight': 0.004016799404500769}, {'token': '\n', 'start_offset': 172, 'end_offset': 173, 'weight': 0.00625060500505452}, {'token': None, 'start_offset': 174, 'end_offset': 174, 'weight': 0.00625060500505452}, {'token': None, 'start_offset': 175, 'end_offset': 175, 'weight': 0.008033039764716074}, {'token': '}', 'start_offset': 176, 'end_offset': 177, 'weight': 0.006258600642188237}, {'token': '\n', 'start_offset': 177, 'end_offset': 178, 'weight': -0.001769257179246499}, {'token': None, 'start_offset': 179, 'end_offset': 179, 'weight': -0.0008674612555366295}, {'token': 'return', 'start_offset': 180, 'end_offset': 186, 'weight': -0.004056363302068069}, {'token': '-', 'start_offset': 187, 'end_offset': 188, 'weight': -0.003237610618368937}, {'token': '1', 'start_offset': 188, 'end_offset': 189, 'weight': 0.0005591937675116917}, {'token': ';', 'start_offset': 189, 'end_offset': 190, 'weight': 0.0005591937675116917}, {'token': '\n', 'start_offset': 190, 'end_offset': 191, 'weight': -0.002472581587827359}, {'token': '}', 'start_offset': 191, 'end_offset': 192, 'weight': -0.002472581587827359}, {'token': '\n', 'start_offset': 192, 'end_offset': 193, 'weight': 0.003798529259764995}], 'node_average_weight': 0.0011110058644922768, 'node_sum_weight': 0.0911024808883667}}

    lineVul_res_list = lineVul_res_list[3:10]
    red_ranges = lineVul_res_list
    black_ranges = node_mapped_linevul_tokens_concat
    print("original_code:\n", original_code)
    print("red_ranges:\n", red_ranges)
    print("black_ranges:\n", black_ranges)
    draw_code_with_underlines_scaled(original_code, red_ranges, black_ranges)



