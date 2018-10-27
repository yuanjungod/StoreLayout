import os
import math
from datetime import datetime
import pandas as pd
from lib.box_analyze_strategy import BoxAnalyseStrategy
from lib.data_loader import DataLoader
import random
import copy
from lib.config import *


class LayoutManager(object):
    Man_upper_body_str = "男上款数"
    Man_pants_str = "男下款数"
    Woman_upper_body_str = "女上款数"
    Women_pants_str = "女下款数"

    Upper_body_category_list = ["上衣", "套装"]
    Pants_category_list = ["裤子", "下装", "裤装"]
    Entire_body_list = ["套装"]

    Import_rate = {
        1: 1,
        2: 3 / 5,
        3: 2 / 5,
        4: 1 / 5,
        5: 1 / 5
    }

    def __init__(self, shop_property_path):
        self.box_analyse = BoxAnalyseStrategy()
        self.category_sort_info = dict()
        self.shop_property_dict = dict()
        print("数据分析中......")
        self.box_analyse_result, category_sort_info = self.box_analyse.analyse(
            os.getcwd() + "/data/details/20181024/utf8_files/")
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
            category_sort_info[i] = [[j[0] / a_sum, j[1]] for j in category_sort_info[i]]
            for item in category_sort_info[i]:
                if item[1][0] not in self.category_sort_info[i]:
                    self.category_sort_info[i][item[1][0]] = dict()
                self.category_sort_info[i][item[1][0]][item[1][1]] = item[0]
        for i in self.category_sort_info:
            confidence = 10 * sum([len(self.category_sort_info[i][k]) for k in self.category_sort_info[i]])
            # confidence = 1
            for j in self.category_sort_info[i]:
                for h in self.category_sort_info[i][j]:
                    self.category_sort_info[i][j][h] *= confidence
                # print(i, j, self.category_sort_info[i][j])
        shop_pd = pd.read_csv(shop_property_path)
        for i in shop_pd.index:
            shop_property = shop_pd.loc[i]
            self.shop_property_dict[shop_property[0]] = dict()
            self.shop_property_dict[shop_property[0]][self.Man_upper_body_str] = shop_property[3]
            self.shop_property_dict[shop_property[0]][self.Woman_upper_body_str] = shop_property[5]
            self.shop_property_dict[shop_property[0]][self.Man_pants_str] = shop_property[4]
            self.shop_property_dict[shop_property[0]][self.Women_pants_str] = shop_property[6]

    @classmethod
    def section_choose(cls, position, result_pd):
        # print("position", position)
        index_cell_length = result_pd.shape[0] // INDEX_DIVIDE
        column_cell_length = result_pd.shape[1] // COLUMN_DIVIDE
        cash_orientation = BoxAnalyseStrategy.get_value_index_column(result_pd, 10009)
        if sum([i[1] for i in cash_orientation]) / len(cash_orientation) < len(result_pd.loc[0]) / 2:
            return max(INDEX_DIVIDE - math.ceil((position[0] + 0.01) / index_cell_length), 0), \
                   min(math.ceil((position[1] + 0.01) / column_cell_length) - 1, COLUMN_DIVIDE - 1)
        else:
            return max(INDEX_DIVIDE - math.ceil((position[0] + 0.01) / index_cell_length), 0), \
                   max(COLUMN_DIVIDE - math.ceil((position[1] + 0.01) / column_cell_length), 0)

    # MODEL NEED
    @classmethod
    def internal_sort(cls, clothing, context):
        clothing.sort(key=lambda a: sum([a["category_score_list"][i] * (10 ** 9 / (1000 ** i)) for i in range(
            len(a["category_score_list"]))]) - 20 ** math.log((datetime.now() - a["date"]).days), reverse=True)
        # print(clothing)
        return clothing

    def choose_best(self, section, boy_clothing, girl_clothing, context, rank=0):
        if section[1] < COLUMN_DIVIDE // 2 and sum([i["count"] for i in boy_clothing]) > 0:
            clothing = boy_clothing
        elif section[1] >= COLUMN_DIVIDE // 2 and sum([i["count"] for i in girl_clothing]) > 0:
            clothing = girl_clothing
        else:
            clothing = boy_clothing if sum([i["count"] for i in boy_clothing]) > 0 else girl_clothing
        for i in clothing:
            i["category_score_list"] = [self.category_sort_info[j].get(section[0], {}).get(section[1], 0) for j in
                                        i["category"]]

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

    def get_layout_clothing(self, plan):
        print("before plan", plan)
        layout_clothing_dict = dict()
        star_clothing_dict = dict()
        display_clothing = copy.deepcopy(self.data_loader.display_clothing_dict)
        for i in display_clothing:
            if i.find("*") != -1:
                star_clothing_dict[i] = display_clothing[i]
        for i in star_clothing_dict:
            display_clothing.pop(i)

        for key, value in star_clothing_dict.items():
            star = value[DataLoader.Position_str][0]
            have_star_props = False
            for i in self.data_loader.props_dict:
                for j in self.data_loader.props_dict[i]:
                    if "*" not in self.data_loader.props_dict[i][j]:
                        continue
                    # print(star, self.data_loader.props_dict[i][j]["*"])
                    if value[DataLoader.Sex_str] == 1 and "男" not in star:
                        star = "男" + star
                    elif value[DataLoader.Sex_str] == 0 and "女" not in star:
                        star = "女" + star
                    # print(star, self.data_loader.props_dict[i][j]["*"])
                    if star == self.data_loader.props_dict[i][j]["*"]:
                        # print("fu")
                        have_star_props = True

            if have_star_props:
                if value[DataLoader.Category_str][0] in self.Pants_category_list and value[DataLoader.Sex_str] == 1:
                    # print("fu11", value[DataLoader.Style_str])
                    plan[self.Man_pants_str] -= value[DataLoader.Style_str][0]
                elif value[DataLoader.Category_str][0] in self.Pants_category_list and value[DataLoader.Sex_str] == 0:
                    # print("fu12", value[DataLoader.Style_str])
                    plan[self.Women_pants_str] -= value[DataLoader.Style_str][0]
                elif value[DataLoader.Category_str][0] in self.Upper_body_category_list and value[
                    DataLoader.Sex_str] == 0:
                    # print("fu13", value[DataLoader.Style_str])
                    plan[self.Woman_upper_body_str] -= value[DataLoader.Style_str][0]
                elif value[DataLoader.Category_str][0] in self.Upper_body_category_list and value[
                    DataLoader.Sex_str] == 1:
                    # print("fu14", value[DataLoader.Style_str])
                    plan[self.Man_upper_body_str] -= value[DataLoader.Style_str][0]

        print("after plan", plan)

        display_clothing_list = list()
        for i in display_clothing:
            display_clothing[i]["count"] = 0
            display_clothing[i]["name"] = i

            root_category = display_clothing[i][DataLoader.Category_str][0]
            sex = display_clothing[i][DataLoader.Sex_str]
            display_clothing[i][self.Man_upper_body_str] = 1 \
                if root_category in self.Upper_body_category_list and sex == 1 else 0
            display_clothing[i][self.Woman_upper_body_str] = 1 \
                if root_category in self.Upper_body_category_list and sex == 0 else 0
            display_clothing[i][self.Man_pants_str] = 1 \
                if root_category in self.Pants_category_list and sex == 1 else 0
            display_clothing[i][self.Women_pants_str] = 1 \
                if root_category in self.Pants_category_list and sex == 0 else 0
            display_clothing_list.append(display_clothing[i])

        print(plan)
        first_circle = True
        while sum(plan.values()) > 0:
            print("fuck!!!!!fuck!!!!!fuck!!!!!", sum(plan.values()), plan[self.Man_upper_body_str],
                  plan[self.Woman_upper_body_str], plan[self.Man_pants_str], plan[self.Women_pants_str])

            for i in display_clothing_list:
                # base_count = i[DataLoader.Style_str][0] if i[DataLoader.Style_str][0] > 0 else i[DataLoader.Style_str][1]
                base_count = int((i[DataLoader.Style_str][0] + i[DataLoader.Style_str][1])/2)
                i["threshold_value"] = min(math.ceil(self.Import_rate[i[DataLoader.Importance_str]] * base_count), 3)

                # if i["count"] + i["threshold_value"] > 8:
                #     i["threshold_value"] = 0
                if i["count"] + i["threshold_value"] > 6 and i["threshold_value"] > 0:
                    i["threshold_value"] = 1

                # print(i["threshold_value"], i["name"], i[DataLoader.Style_str], i["count"] + i["threshold_value"])
                if i[DataLoader.Importance_str] <= 2 and first_circle is True:
                    if i["name"].find("￥") != -1:
                        second_props_num = self.data_loader.second_props_num_dict.get(i["name"].split("￥")[-1], 0)
                        print("second_props_num", second_props_num, i["name"])
                        if second_props_num > i["threshold_value"]:
                            i["threshold_value"] = min(second_props_num, base_count)

                # print("threshold_value", i["threshold_value"], i["name"])
                i["sort_important"] = i[DataLoader.Importance_str]

            first_circle = False

            while plan[self.Man_upper_body_str] > 0:

                display_clothing_list.sort(
                    key=lambda a: a[self.Man_upper_body_str] * (10 ** 11) + min(a["threshold_value"], 1) * (
                            10 ** 10) + (5 - a["sort_important"]) * (10 ** 9), reverse=True)
                a = None
                for i in range(len(display_clothing_list)):
                    if (display_clothing_list[i][DataLoader.Style_str][0] > 0 or
                        display_clothing_list[i][DataLoader.Style_str][1] > 0) and \
                            display_clothing_list[i][self.Man_upper_body_str] == 1 and \
                            display_clothing_list[i]["threshold_value"] > 0:
                        a = display_clothing_list[i]
                        if a[DataLoader.Style_str][0] > 0:
                            a[DataLoader.Style_str][0] -= 1
                            a[DataLoader.Style_str][1] -= 1
                        elif a[DataLoader.Style_str][1] > 0:
                            a[DataLoader.Style_str][1] -= 1
                        if a[DataLoader.Style_str][0] <= 0 and a[DataLoader.Style_str][1] < 5:
                            a["sort_important"] += 1

                        a["count"] += 1
                        plan[self.Man_upper_body_str] -= 1
                        a["threshold_value"] -= 1
                        # print("break1")
                        break
                # print("break2", a)
                if a is None:
                    # print("break3")
                    break

            while plan[self.Man_pants_str] > 0:
                display_clothing_list.sort(key=lambda a: a[self.Man_pants_str] * (10 ** 11) + min(a["threshold_value"], 1) * (
                        10 ** 10) + (5 - a["sort_important"]) * (10 ** 9), reverse=True)

                a = None
                for i in range(len(display_clothing_list)):
                    if (display_clothing_list[i][DataLoader.Style_str][0] > 0 or
                        display_clothing_list[i][DataLoader.Style_str][1] > 0) and \
                            display_clothing_list[i][self.Man_pants_str] == 1 and \
                            display_clothing_list[i]["threshold_value"] > 0:
                        a = display_clothing_list[i]
                        if a[DataLoader.Style_str][0] > 0:
                            a[DataLoader.Style_str][0] -= 1
                            a[DataLoader.Style_str][1] -= 1
                        elif a[DataLoader.Style_str][1] > 0:
                            a[DataLoader.Style_str][1] -= 1
                        if a[DataLoader.Style_str][0] <= 0 and a[DataLoader.Style_str][1] < 5:
                            a["sort_important"] += 1

                        a["count"] += 1
                        plan[self.Man_pants_str] -= 1
                        a["threshold_value"] -= 1
                        break
                if a is None:
                    break

            while plan[self.Woman_upper_body_str] > 0:
                display_clothing_list.sort(
                    key=lambda a: a[self.Woman_upper_body_str] * (10 ** 11) + min(a["threshold_value"], 1) * (
                            10 ** 10) + (5 - a["sort_important"]) * (10 ** 9), reverse=True)

                a = None
                for i in range(len(display_clothing_list)):
                    if (display_clothing_list[i][DataLoader.Style_str][0] > 0 or
                        display_clothing_list[i][DataLoader.Style_str][1] > 0) and \
                            display_clothing_list[i][self.Woman_upper_body_str] == 1 and \
                            display_clothing_list[i]["threshold_value"] > 0:
                        a = display_clothing_list[i]
                        if a[DataLoader.Style_str][0] > 0:
                            a[DataLoader.Style_str][0] -= 1
                            a[DataLoader.Style_str][1] -= 1
                        elif a[DataLoader.Style_str][1] > 0:
                            a[DataLoader.Style_str][1] -= 1
                        if a[DataLoader.Style_str][0] <= 0 and a[DataLoader.Style_str][1] < 5:
                            a["sort_important"] += 1

                        a["count"] += 1
                        plan[self.Woman_upper_body_str] -= 1
                        a["threshold_value"] -= 1
                        break
                if a is None:
                    break

            while plan[self.Women_pants_str] > 0:
                display_clothing_list.sort(key=lambda a: a[self.Women_pants_str] * (10 ** 11) + min(a["threshold_value"], 1) * (
                        10 ** 10) + (5 - a["sort_important"]) * (10 ** 9), reverse=True)

                a = None
                for i in range(len(display_clothing_list)):
                    if (display_clothing_list[i][DataLoader.Style_str][0] > 0 or
                        display_clothing_list[i][DataLoader.Style_str][1] > 0) and \
                            display_clothing_list[i][self.Women_pants_str] == 1 and \
                            display_clothing_list[i]["threshold_value"] > 0:
                        a = display_clothing_list[i]
                        if a[DataLoader.Style_str][0] > 0:
                            a[DataLoader.Style_str][0] -= 1
                            a[DataLoader.Style_str][1] -= 1
                        elif a[DataLoader.Style_str][1] > 0:
                            a[DataLoader.Style_str][1] -= 1
                        if a[DataLoader.Style_str][0] <= 0 and a[DataLoader.Style_str][1] < 5:
                            a["sort_important"] += 1

                        a["count"] += 1
                        plan[self.Women_pants_str] -= 1
                        a["threshold_value"] -= 1
                        break
                if a is None:
                    break

        for i in display_clothing_list:
            layout_clothing_dict[i["name"]] = i
        return layout_clothing_dict, star_clothing_dict

    def allocation_cell_count(self, layout_clothing_dict, total_cell_count):
        mean_cell_style_count = sum([value["count"] for key, value in layout_clothing_dict.items()]) / total_cell_count
        current_mean_cell_style_count = mean_cell_style_count
        allocation_cell_dict = {}

        values = list(layout_clothing_dict.values())
        values.sort(key=lambda a: a["count"], reverse=True)

        current_cell = 0
        for value in values:
            if "current_cell_num" not in value:
                value["current_cell_num"] = 0
            if current_cell < total_cell_count:
                need_cells = math.ceil(value["count"] / current_mean_cell_style_count + 0.5)
                value["current_cell_num"] = need_cells
                for i in range(need_cells):
                    if current_cell + i >= total_cell_count:
                        break
                    if current_cell + i not in allocation_cell_dict:
                        allocation_cell_dict[current_cell + i] = list()
                    allocation_cell_dict[current_cell + i].append([value["name"]])
                current_cell += need_cells
        for i in allocation_cell_dict:
            print("allocation", allocation_cell_dict[i], len(allocation_cell_dict))

        for value in values:
            if value["current_cell_num"] == 0:
                print("not be set", value["name"], value["count"])

    def layout_special(self):
        for i in self.data_loader.props_dict:
            for j in self.data_loader.props_dict[i]:
                if "*" in self.data_loader.props_dict[i][j]:
                    print("djdjdjdj", self.data_loader.props_dict[i][j]["*"])
                    for name in self.data_loader.display_clothing_dict:
                        print(name, self.data_loader.display_clothing_dict[name][DataLoader.Sex_str] == 0)
                        props_full_name = name.split("*")[-1]
                        if name.find("*") != -1 and self.data_loader.display_clothing_dict[name][DataLoader.Sex_str] == 1:
                            props_full_name = "男" + props_full_name
                        elif name.find("*") != -1 and self.data_loader.display_clothing_dict[name][DataLoader.Sex_str] == 0:
                            props_full_name = "女" + props_full_name
                        if props_full_name == self.data_loader.props_dict[i][j]["*"]:
                            # content = [name.split["*"][0] for i in range(
                            #     len(self.data_loader.encode_pd.at[i, j]))]
                            self.data_loader.result_pd.at[i, j] = "/".join([name.split("*")[0] for k in range(
                                len(self.data_loader.encode_pd.loc[i, j]))]) + "*" + props_full_name
                elif "￥" in self.data_loader.props_dict[i][j]:
                    for name in self.data_loader.display_clothing_dict:
                        props_full_name = name.split("*")[-1]
                        if props_full_name == self.data_loader.props_dict[i][j]["￥"]:
                            self.data_loader.result_pd.at[i, j] = "/".join([name.split("￥")[0] for k in range(
                                len(self.data_loader.encode_pd.at[i, j]))]) + "￥" + props_full_name
                            if "count" not in self.data_loader.display_clothing_dict[name]:
                                self.data_loader.display_clothing_dict[name]["count"] = 0
                            self.data_loader.display_clothing_dict[name]["count"] += 1
        print(self.data_loader.result_pd)
        # exit()

    def layout(self, csv_path, plan_path, context=None):
        self.data_loader.load_csv(csv_path, plan_path)
        self.layout_special()
        total_cell = 0
        for i in self.data_loader.result_pd.index:
            for j in range(len(self.data_loader.result_pd.loc[i])):
                if self.data_loader.result_pd.at[i, j] == -1 and \
                        "*" not in self.data_loader.props_dict[i][j] and \
                        "￥" not in self.data_loader.props_dict[i][j]:
                    a = self.data_loader.encode_pd.at[i, j]
                    for k in a:
                        if isinstance(k, list):
                            for h in k:
                                if h != 129:
                                    total_cell += 1
                        else:
                            if k != 129:
                                total_cell += 1

        print("total_cell", total_cell, csv_path)

        a, star = self.get_layout_clothing(self.shop_property_dict[csv_path.split("/")[-1].split(".")[0]])

        print(sum([a[i]["count"] for i in a]))

        # for i in a:
        #     print("fuck count", a[i]["count"], a[i]["name"])

        cell_capacity = sum([a[i]["count"] for i in a]) / total_cell

        # for i in a:
        #     print("fuck count", a[i]["count"], a[i]["count"] / cell_capacity)
        print("sum", sum([int(a[i]["count"] / cell_capacity) for i in a]))

        print(layout.data_loader.display_clothing_dict, cell_capacity)

        self.allocation_cell_count(a, total_cell)

        if sum(layout.data_loader.display_clothing_dict.values()) != total_cell:
            print("%s店,需要摆放%s个单元格,提供了%s单元格服装,将自动生成摆放的服装,请知悉" % (
            csv_path.split("/")[-1].split(".")[0], total_cell, sum(layout.data_loader.display_clothing_dict.values())))
            # print(layout.data_loader.display_clothing_dict)
            probability_dict = copy.deepcopy(layout.data_loader.display_clothing_dict)
            total = sum(layout.data_loader.display_clothing_dict.values())
            keys = list(probability_dict.keys())
            for i in range(len(keys)):
                if i > 0:
                    probability_dict[keys[i]] = probability_dict[keys[i - 1]] + probability_dict[keys[i]] / total
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
            index_length = len(self.data_loader.result_pd.index) - 1
            fix_i_list = [i, len(self.data_loader.result_pd.loc[0]) - i - 1]

            for fix_i in set(fix_i_list):
                for j in self.data_loader.result_pd.index:
                    if self.data_loader.result_pd.at[index_length - j, fix_i] not in [-1]:
                        continue

                    def get_result_list(index, column):
                        section = self.section_choose([index_length - j, fix_i], self.data_loader.result_pd)
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

                    self.data_loader.result_pd.at[index_length - j, fix_i] = get_result_list(index_length - j, fix_i)

                    small_search_fix_i_neighbor = fix_i - 1
                    while small_search_fix_i_neighbor > 0 and \
                            self.data_loader.result_pd.at[index_length - j, small_search_fix_i_neighbor] in [-1]:
                        self.data_loader.result_pd.at[index_length - j, small_search_fix_i_neighbor] = get_result_list(
                            index_length - j, small_search_fix_i_neighbor)
                        small_search_fix_i_neighbor -= 1

                    big_search_fix_i_neighbor = fix_i + 1
                    while big_search_fix_i_neighbor < len(self.data_loader.result_pd.loc[0]) - 2 and \
                            self.data_loader.result_pd.at[index_length - j, big_search_fix_i_neighbor] in [-1]:
                        self.data_loader.result_pd.at[index_length - j, big_search_fix_i_neighbor] = get_result_list(
                            index_length - j, big_search_fix_i_neighbor)
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
        self.data_loader.result_pd.to_csv(
            os.getcwd() + "/data/result/" + "%s.csv" % csv_path.split("/")[-1].split(".")[0])
        print("保存至[%s" % os.getcwd() + "/data/result/" + "%s.csv]" % csv_path.split("/")[-1].split(".")[0])


if __name__ == "__main__":
    shop_property_csv_path = "/Users/quantum/code/StoreLayout/data/details/20181024/shop_property_utf8.csv"
    layout = LayoutManager(shop_property_csv_path)
    root_path = "/Users/quantum/code/StoreLayout/data/details/20181024/utf8_files/"
    plan_path = "/Users/quantum/code/StoreLayout/data/plan/2018_15_plan_utf8.csv"
    for file_path in os.listdir(root_path):
        layout.layout(os.path.join(root_path, file_path), plan_path, CONTEXT_DICT)
