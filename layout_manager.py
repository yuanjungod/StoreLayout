import os
import math
from lib.box_analyze_strategy import BoxAnalyseStragy
from lib.data_loader import DataLoader
from lib.config import *


class LayoutManager(object):

    def __init__(self):
        self.box_analyse = BoxAnalyseStragy()
        self.category_sort_info = dict()
        self.box_analyse_result, category_sort_info = self.box_analyse.analyse(
            os.getcwd() + "/data/details/", 4)
        self.data_loader = DataLoader()
        for i in self.box_analyse_result:
            for j in self.box_analyse_result[i]:
                self.box_analyse_result[i][j]['date'].sort(reverse=True)
                print("date", i, j, self.box_analyse_result[i][j]['date'])
                print("season", i, j, self.box_analyse_result[i][j]['season'])
                print("sex", i, j, self.box_analyse_result[i][j]['sex'])
                print("category", i, j, self.box_analyse_result[i][j]['category'])
        for i in category_sort_info:
            self.category_sort_info[i] = dict()
            a_sum = sum([k[0] for k in category_sort_info[i]])
            category_sort_info[i] = [[j[0]/a_sum, j[1]] for j in category_sort_info[i]]
            for item in category_sort_info[i]:
                if item[1][0] not in self.category_sort_info[i]:
                    self.category_sort_info[i][item[1][0]] = dict()
                self.category_sort_info[i][item[1][0]][item[1][1]] = item[0]
        for i in self.category_sort_info:
            for j in self.category_sort_info[i]:
                print(i, j, self.category_sort_info[i][j])

    @classmethod
    def section_choose(cls, position, result_pd):
        cash_orientation = BoxAnalyseStragy.get_value_index_column(result_pd, 10009)
        if sum([i[1] for i in cash_orientation]) / len(cash_orientation) < len(result_pd.loc[0]) / 2:
            return min(math.ceil(position[0] / 4) - 1, 3), min(math.ceil(position[1] / 4) - 1, 3)
        else:
            return min(math.ceil(position[0] / 4) - 1, 3), max(3-math.ceil(position[1] / 4) + 1, 0)

    def get_choose_best(self, section, boy_clothing, girl_clothing, rank=0):
        if section[1] < 2 and sum([i["count"] for i in boy_clothing]) > 0:
            clothing = boy_clothing
        elif section[1] >= 2 and sum([i["count"] for i in girl_clothing]) > 0:
            clothing = girl_clothing
        else:
            clothing = boy_clothing if sum([i["count"] for i in boy_clothing]) > 0 else girl_clothing
        for i in clothing:
            i["category_score_list"] = [self.category_sort_info[j].get(section[0], {}).get(section[1], 0) for j in i["category"]]

        clothing.sort(key=lambda a: a["category_score_list"][0], reverse=True)
        if rank > len(clothing):
            best_one = clothing[-1]
        else:
            print(clothing, rank)
            best_one = clothing[rank]

        if best_one["count"] > 1:
            best_one["count"] -= 1
        else:
            clothing.remove(best_one)

        return best_one['index']

    def layout(self, csv_path):
        self.data_loader.load_csv(csv_path)
        total_cell = 0
        for i in self.data_loader.result_pd.index:
            for j in range(len(self.data_loader.result_pd.loc[i])):
                print(i, j, self.data_loader.encode_pd.shape, self.data_loader.result_pd.shape, len(self.data_loader.result_pd.loc[i]))
                if self.data_loader.result_pd.at[i, j] == -1:
                    a = self.data_loader.encode_pd.at[i, j]
                    for k in a:
                        if isinstance(k, list):
                            for h in k:
                                total_cell += 1
                        else:
                            total_cell += 1
        if sum(layout.data_loader.display_clothing_dict.values()) != total_cell:
            print("可以摆放%s个单元格，提供了%s单元格服装,需要调整" % (total_cell, sum(layout.data_loader.display_clothing_dict.values())))
            exit()
        cash_orientation = BoxAnalyseStragy.get_value_index_column(self.data_loader.result_pd, 10009)
        boy_clothing = list()
        girl_clothing = list()
        for key, value in layout.data_loader.display_clothing_dict.items():
            clothing_info = layout.data_loader.name_dict[key]
            if clothing_info["sex"] == 1:
                clothing_info["count"] = value
                boy_clothing.append(clothing_info)
            else:
                clothing_info["count"] = value
                girl_clothing.append(clothing_info)
        boy_clothing.sort(key=lambda a: a["date"], reverse=True)
        girl_clothing.sort(key=lambda a: a["date"], reverse=True)

        print("boy_clothing", boy_clothing)
        print("girl_clothing", girl_clothing)
        print("layout begin!!!!")

        for i in range(len(self.data_loader.result_pd.loc[0])):
            index_length = len(self.data_loader.result_pd.index)-1
            for j in self.data_loader.result_pd.index:
                if self.data_loader.result_pd.at[index_length-j, i] not in [-1]:
                    continue
                section = self.section_choose([index_length-j, i], self.data_loader.result_pd)
                result_list = self.data_loader.encode_pd.at[index_length-j, i]
                print(result_list)
                for k in range(len(result_list)):
                    if isinstance(result_list[k], list):
                        for h in range(len(result_list[k])):
                            print(index_length-j, i)
                            result_list[k][h] = self.get_choose_best(section, boy_clothing, girl_clothing, 0)

                    else:
                        print(index_length-j, i)
                        result_list[k] = self.get_choose_best(section, boy_clothing, girl_clothing, 0)
                self.data_loader.result_pd.at[index_length-j, i] = result_list

        self.data_loader.result_pd.to_csv("result1.csv")
        for i in self.data_loader.result_pd.index:
            for j in range(len(self.data_loader.result_pd.loc[i])):
                a = self.data_loader.result_pd.at[i, j]
                if not isinstance(a, list):
                    if a in self.data_loader.category_dict and a != 0:
                        a = self.data_loader.category_dict[a]["name"]
                    elif a in entity_dict.values():
                        for key, value in entity_dict.items():
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
        print(self.data_loader.result_pd)
        self.data_loader.result_pd.to_csv("result.csv")


if __name__ == "__main__":
    layout = LayoutManager()
    layout.layout("/Users/happy/code/StoreLayout/data/details/高县.csv")








