import os
import math
from datetime import datetime
from lib.box_analyze_strategy import BoxAnalyseStrategy
from lib.data_loader import DataLoader
import random
import copy
from lib.config import *


class LayoutManager(object):

    def __init__(self):
        self.box_analyse = BoxAnalyseStrategy()
        self.category_sort_info = dict()
        print("数据分析中......")
        self.box_analyse_result, category_sort_info = self.box_analyse.analyse(
            os.getcwd() + "/data/details/")
        print("分析完成")
        self.data_loader = DataLoader()
        for i in self.box_analyse_result:
            for j in self.box_analyse_result[i]:
                self.box_analyse_result[i][j]['date'].sort(reverse=True)
                # print("date", i, j, self.box_analyse_result[i][j]['date'])
                # print("season", i, j, self.box_analyse_result[i][j]['season'])
                # print("sex", i, j, self.box_analyse_result[i][j]['sex'])
                # print("category", i, j, self.box_analyse_result[i][j]['category'])
        for i in category_sort_info:
            self.category_sort_info[i] = dict()
            a_sum = sum([k[0] for k in category_sort_info[i]])
            category_sort_info[i] = [[j[0]/a_sum, j[1]] for j in category_sort_info[i]]
            for item in category_sort_info[i]:
                if item[1][0] not in self.category_sort_info[i]:
                    self.category_sort_info[i][item[1][0]] = dict()
                self.category_sort_info[i][item[1][0]][item[1][1]] = item[0]
        for i in self.category_sort_info:
            confidence = 10*sum([len(self.category_sort_info[i][k]) for k in self.category_sort_info[i]])
            # confidence = 1
            for j in self.category_sort_info[i]:
                for h in self.category_sort_info[i][j]:
                    self.category_sort_info[i][j][h] *= confidence
                # print(i, j, self.category_sort_info[i][j])

    @classmethod
    def section_choose(cls, position, result_pd):
        # print("position", position)
        index_cell_length = result_pd.shape[0]//INDEX_DIVIDE
        column_cell_length = result_pd.shape[1]//COLUMN_DIVIDE
        cash_orientation = BoxAnalyseStrategy.get_value_index_column(result_pd, 10009)
        if sum([i[1] for i in cash_orientation]) / len(cash_orientation) < len(result_pd.loc[0]) / 2:
            return max(INDEX_DIVIDE - math.ceil((position[0]+0.01) / index_cell_length), 0), \
                   min(math.ceil((position[1]+0.01) / column_cell_length)-1, COLUMN_DIVIDE - 1)
        else:
            return max(INDEX_DIVIDE - math.ceil((position[0]+0.01) / index_cell_length), 0), \
                   max(COLUMN_DIVIDE-math.ceil((position[1]+0.01) / column_cell_length), 0)

    # MODEL NEED
    @classmethod
    def internal_sort(cls, clothing, context):
        clothing.sort(key=lambda a: sum([a["category_score_list"][i]*(10**9/(1000**i)) for i in range(
            len(a["category_score_list"]))])-20**math.log((datetime.now() - a["date"]).days), reverse=True)
        # print(clothing)
        return clothing

    def choose_best(self, section, boy_clothing, girl_clothing, context, rank=0):
        if section[1] < COLUMN_DIVIDE//2 and sum([i["count"] for i in boy_clothing]) > 0:
            clothing = boy_clothing
        elif section[1] >= COLUMN_DIVIDE//2 and sum([i["count"] for i in girl_clothing]) > 0:
            clothing = girl_clothing
        else:
            clothing = boy_clothing if sum([i["count"] for i in boy_clothing]) > 0 else girl_clothing
        for i in clothing:
            i["category_score_list"] = [self.category_sort_info[j].get(section[0], {}).get(section[1], 0) for j in i["category"]]

        clothing = self.internal_sort(clothing, context)
        if rank >= len(clothing):
            best_one = clothing[-1]
        else:
            best_one = clothing[rank]

        if best_one["count"] > 1:
            best_one["count"] -= 1
        else:
            clothing.remove(best_one)

        return best_one['index']

    def layout(self, csv_path, context=None):
        self.data_loader.load_csv(csv_path)
        total_cell = 0
        for i in self.data_loader.result_pd.index:
            for j in range(len(self.data_loader.result_pd.loc[i])):
                if self.data_loader.result_pd.at[i, j] == -1:
                    a = self.data_loader.encode_pd.at[i, j]
                    for k in a:
                        if isinstance(k, list):
                            for h in k:
                                if h != 129:
                                    total_cell += 1
                        else:
                            if k != 129:
                                total_cell += 1
        if sum(layout.data_loader.display_clothing_dict.values()) != total_cell:
            print("%s店,需要摆放%s个单元格,提供了%s单元格服装,将自动生成摆放的服装,请知悉" % (csv_path.split("/")[-1].split(".")[0], total_cell, sum(layout.data_loader.display_clothing_dict.values())))
            # print(layout.data_loader.display_clothing_dict)
            probability_dict = copy.deepcopy(layout.data_loader.display_clothing_dict)
            total = sum(layout.data_loader.display_clothing_dict.values())
            keys = list(probability_dict.keys())
            for i in range(len(keys)):
                if i > 0:
                    probability_dict[keys[i]] = probability_dict[keys[i-1]] + probability_dict[keys[i]]/total
                else:
                    probability_dict[keys[i]] = probability_dict[keys[i]] / total
                layout.data_loader.display_clothing_dict[keys[i]] = 0
            print("probability_dict", probability_dict)
            current_total_cell = total_cell
            while current_total_cell > 0:
                # print("fuck")
                random_value = random.random()
                for i in range(len(keys)):
                    if random_value < probability_dict[keys[i]]:
                        layout.data_loader.display_clothing_dict[keys[i]] += 1
                        break
                current_total_cell -= 1
            print("display_clothing_dict", layout.data_loader.display_clothing_dict)

        boy_clothing = list()
        girl_clothing = list()
        for key, value in layout.data_loader.display_clothing_dict.items():
            if value < 0:
                continue
            clothing_info = layout.data_loader.name_dict[key]

            if clothing_info["sex"] == 1:
                clothing_info["count"] = value
                clothing_info["name"] = key
                boy_clothing.append(clothing_info)
            else:
                clothing_info["count"] = value
                clothing_info["name"] = key
                girl_clothing.append(clothing_info)

        print("%s店,生成服装布局..." % csv_path.split("/")[-1].split(".")[0])

        for i in range(len(self.data_loader.result_pd.loc[0])):
            index_length = len(self.data_loader.result_pd.index)-1
            fix_i_list = [i, len(self.data_loader.result_pd.loc[0]) - i - 1]

            for fix_i in set(fix_i_list):
                for j in self.data_loader.result_pd.index:
                    if self.data_loader.result_pd.at[index_length-j, fix_i] not in [-1]:
                        continue

                    def get_result_list(index, column):
                        section = self.section_choose([index_length-j, fix_i], self.data_loader.result_pd)
                        # print("section", section)
                        result_list = self.data_loader.encode_pd.at[index, column]
                        rank = 0
                        for k in range(len(result_list)):
                            if isinstance(result_list[k], list):
                                for h in range(len(result_list[k])):
                                    if result_list[k][h] != 129:
                                        result_list[k][h] = self.choose_best(
                                            section, boy_clothing, girl_clothing, context, rank)
                                        rank += 1

                            else:
                                result_list[k] = self.choose_best(section, boy_clothing, girl_clothing, context, rank)
                                rank += 1
                        return result_list

                    self.data_loader.result_pd.at[index_length-j, fix_i] = get_result_list(index_length-j, fix_i)

                    small_search_fix_i_neighbor = fix_i-1
                    while small_search_fix_i_neighbor > 0 and \
                            self.data_loader.result_pd.at[index_length-j, small_search_fix_i_neighbor] in [-1]:
                        self.data_loader.result_pd.at[index_length - j, small_search_fix_i_neighbor] = get_result_list(
                            index_length-j, small_search_fix_i_neighbor)
                        small_search_fix_i_neighbor -= 1

                    big_search_fix_i_neighbor = fix_i + 1
                    while big_search_fix_i_neighbor < len(self.data_loader.result_pd.loc[0])-2 and \
                            self.data_loader.result_pd.at[index_length - j, big_search_fix_i_neighbor] in [-1]:
                        self.data_loader.result_pd.at[index_length - j, big_search_fix_i_neighbor] = get_result_list(
                            index_length-j, big_search_fix_i_neighbor)
                        big_search_fix_i_neighbor += 1

        for i in range(len(self.data_loader.result_pd.index)):

            for j in range(len(self.data_loader.result_pd.loc[i])):
                a = self.data_loader.result_pd.at[i, j]
                if not isinstance(a, list):
                    if a in self.data_loader.category_dict and a != 0:
                        a = self.data_loader.category_dict[a]["name"]
                    elif a in ENTITY_DICT.values():
                        for key, value in ENTITY_DICT.items():
                            if a == value:
                                a = key
                else:
                    for k in range(len(a)):
                        if not isinstance(a[k], list):
                            a[k] = self.data_loader.category_dict[a[k]]["name"]
                        else:
                            for h in range(len(a[k])):
                                a[k][h] = self.data_loader.category_dict[a[k][h]]["name"]
                if isinstance(a, list):
                    if len(a) > 1:
                        a = "/".join(["&".join(i) for i in a])
                    else:
                        a = "&".join(a[0])
                self.data_loader.result_pd.at[i, j] = a
        # print(self.data_loader.result_pd)
        self.data_loader.result_pd.to_csv(os.getcwd() + "/data/result/" + "%s.csv" % csv_path.split("/")[-1].split(".")[0])
        print("保存至[%s" % os.getcwd() + "/data/result/" + "%s.csv]" % csv_path.split("/")[-1].split(".")[0])


if __name__ == "__main__":
    layout = LayoutManager()
    root_path = "/Users/quantum/code/StoreLayout/data/details"
    for file_path in os.listdir(root_path):
        layout.layout(os.path.join(root_path, file_path), CONTEXT_DICT)








