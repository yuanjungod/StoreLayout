import os
import time
from .config import *
from .data_loader import DataLoader


class BoxAnalyseStragy(object):

    def __init__(self):
        self.data_list = None
        self.box_gather = dict()

    @classmethod
    def get_value_index_column(cls, encode_pd, value):
        result_list = list()
        for i in encode_pd.index:
            for j in range(len(encode_pd.loc[i])):
                if value == encode_pd.at[i, j]:
                    result_list.append((i, j))
        return result_list

    @classmethod
    def get_divide_partition(cls, total_length, divide_count=4):
        result_list = list()
        unit_length = total_length // (divide_count + 1)
        for i in range(divide_count + 1):
            result_list.append(i * unit_length)
        result_list.append(total_length)
        return result_list

    def get_all_data(self, root_path):
        self.data_list = list()
        for i in os.listdir(root_path):
            for j in range(6):
                print('\r#########{},分析中'.format(i, "."*(j+1)), end='')
                time.sleep(0.1)
                # print("###################################", i)
            data = DataLoader()
            data.load_csv(os.path.join(root_path, i))
            self.data_list.append(data)

    def get_all_box(self, root_path):
        result_box_dict = {}
        if self.data_list is None:
            self.get_all_data(root_path)
        for encode_pd in self.data_list:
            index_divide = self.get_divide_partition(encode_pd.encode_pd.shape[0]+1, divide_count=INDEX_DIVIDE)
            column_divide = self.get_divide_partition(encode_pd.encode_pd.shape[1]+1, divide_count=COLUMN_DIVIDE)
            cash_orientation = self.get_value_index_column(encode_pd.encode_pd, 10009)
            if sum([i[1] for i in cash_orientation]) / len(cash_orientation) < len(encode_pd.encode_pd.loc[0]) / 2:
                for index in range(INDEX_DIVIDE):
                    if index not in result_box_dict:
                        result_box_dict[index] = dict()
                    for column in range(COLUMN_DIVIDE):
                        if column not in result_box_dict[index]:
                            result_box_dict[index][column] = list()
                        result_box_dict[index][column].append(
                            encode_pd.encode_pd.loc[index_divide[INDEX_DIVIDE - 1 - index]:
                                                    index_divide[INDEX_DIVIDE - index], column_divide[column]:
                                                                                        column_divide[column + 1]])
            else:
                for index in range(INDEX_DIVIDE):
                    if index not in result_box_dict:
                        result_box_dict[index] = dict()
                    for column in range(COLUMN_DIVIDE):
                        if column not in result_box_dict[index]:
                            result_box_dict[index][column] = list()
                        result_box_dict[index][column].append(
                            encode_pd.encode_pd.loc[index_divide[INDEX_DIVIDE - 1 - index]:
                                                    index_divide[INDEX_DIVIDE - index], column_divide[COLUMN_DIVIDE-column-1]:
                                                                                        column_divide[COLUMN_DIVIDE-column]])
        return result_box_dict

    @classmethod
    def get_item_from_list(cls, a, result_list):
        if not isinstance(a, list):
            result_list.append(a)
        else:
            for i in a:
                cls.get_item_from_list(i, result_list)
        return result_list

    def analyse(self, root_path):
        box_analyse_result = dict()
        result_box_dict = self.get_all_box(root_path)
        category_dict = self.data_list[0].category_dict
        for i in range(INDEX_DIVIDE):
            if i not in box_analyse_result:
                box_analyse_result[i] = dict()
            for j in range(COLUMN_DIVIDE):
                if j not in box_analyse_result[i]:
                    box_analyse_result[i][j] = dict()
                    tmp_pd_list = result_box_dict[i][j]

                    for tmp_pd in tmp_pd_list:
                        # print(tmp_pd)
                        for k in tmp_pd.index:
                            for h in tmp_pd.loc[k]:
                                h1 = list()
                                h1 = self.get_item_from_list(h, h1)
                                for h2 in h1:
                                    if h2 not in category_dict:
                                        continue
                                    category_info = category_dict[h2]
                                    # print(category_info)
                                    for item in ["sex", "season", "date", "category"]:
                                        if item == "category":
                                            if "category" not in box_analyse_result[i][j]:
                                                box_analyse_result[i][j]["category"] = dict()
                                            for category in category_info[item]:
                                                if category not in box_analyse_result[i][j]["category"]:
                                                    box_analyse_result[i][j]["category"][category] = 0
                                                box_analyse_result[i][j]["category"][category] += 1
                                        elif item == "date":
                                            if "date" not in box_analyse_result[i][j]:
                                                box_analyse_result[i][j]["date"] = list()
                                            box_analyse_result[i][j]["date"].append(category_info[item])
                                        elif item == "season":
                                            if "season" not in box_analyse_result[i][j]:
                                                box_analyse_result[i][j]["season"] = dict()
                                            if category_info[item] not in box_analyse_result[i][j]["season"]:
                                                box_analyse_result[i][j]["season"][category_info[item]] = 0
                                            box_analyse_result[i][j]["season"][category_info[item]] += 1
                                        elif item == "sex":
                                            if "sex" not in box_analyse_result[i][j]:
                                                box_analyse_result[i][j]["sex"] = dict()
                                            if category_info[item] not in box_analyse_result[i][j]["sex"]:
                                                box_analyse_result[i][j]["sex"][category_info[item]] = 0
                                            # print(category_info[item])
                                            box_analyse_result[i][j]["sex"][category_info[item]] += 1
        category_sort_info = {}
        for i in box_analyse_result:
            for j in box_analyse_result[i]:
                for category in box_analyse_result[i][j]["category"]:
                    if category not in category_sort_info:
                        category_sort_info[category] = list()
                    category_sort_info[category].append([box_analyse_result[i][j]["category"][category]/sum(
                        box_analyse_result[i][j]["category"].values()), (i, j)])
        for category in category_sort_info:
            category_sort_info[category].sort(key=lambda a: a[0], reverse=True)

        return box_analyse_result, category_sort_info


if __name__ == "__main__":
    box_analyse = BoxAnalyseStragy()
    # print(box_analyse.get_divide_partition(12, 3))
    # result_box_dict = box_analyse.get_all_box(os.getcwd() + "/../data/details/", 4)
    # for i in result_box_dict:
    #     for j in result_box_dict[i]:
    #         print("#####fuck: %s, %s" % (i, j), result_box_dict[i][j])
    # box_analyse.get_all_data(os.getcwd() + "/../data/details/")
    # print(box_analyse.data_list[0].encode_pd.shape)
    # print(box_analyse.data_list[0].encode_pd.loc[23: 25, 10: 12])
    # result_list = list()
    # result_list = BoxAnalyseStragy.get_item_from_list([[[1, [2]]]], result_list)
    # print(result_list)
    box_analyse_result, category_sort_info = box_analyse.analyse(os.getcwd() + "/../data/details/")
    for i in box_analyse_result:
        for j in box_analyse_result[i]:
            print("#############################", i, j)
            print(box_analyse_result[i][j]["sex"])
            print(box_analyse_result[i][j]["category"])
            print(box_analyse_result[i][j]["season"])
    # print("category_sort_info", category_sort_info)
    for category in category_sort_info:
        print(category, category_sort_info[category])
    # print("box_analyse_result", box_analyse_result)

