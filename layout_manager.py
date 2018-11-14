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
    Man_wall_upper_body_str = "墙男上款数"
    Man_island_upper_body_str = "岛男上款数"
    Man_wall_pants_str = "墙男下款数"
    Man_island_pants_str = "岛男下款数"
    Man_wall_suite_str = "墙男套装数"
    Man_island_suite_str = "岛男套装数"
    Woman_wall_upper_body_str = "墙女上款数"
    Woman_island_upper_body_str = "岛女上款数"
    Women_wall_pants_str = "墙女下款数"
    Women_island_pants_str = "岛女下款数"
    Women_wall_suite_str = "墙女套装数"
    Women_island_suite_str = "岛女套装数"

    Upper_body_category_list = ["上衣", "套装"]
    Pants_category_list = ["裤子", "下装", "裤装", "裙子"]
    Entire_body_list = []
    Suit_list = ["套装"]

    Import_rate = {
        1: 1,
        2: 3 / 5,
        3: 2 / 5,
        4: 1 / 4,
        5: 1 / 4
    }

    def __init__(self):
        self.box_analyse = BoxAnalyseStrategy()
        self.category_sort_info = dict()
        self.shop_property_dict = dict()
        print("数据分析中......")
        self.box_analyse_result, category_sort_info = self.box_analyse.analyse(
            os.getcwd() + "/data/details/20181024/utf8_files/")
        print("分析完成")
        self.data_loader = None
        for i in self.box_analyse_result:
            for j in self.box_analyse_result[i]:
                self.box_analyse_result[i][j]['date'].sort(reverse=True)
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

        self.sex_orientation = -1
        self.man_wall_cell_count = 0
        self.women_wall_cell_count = 0
        self.sex_bit_map = dict()

    def init(self):
        self.man_wall_cell_count = 0
        self.women_wall_cell_count = 0
        self.sex_orientation = -1
        self.sex_bit_map = dict()
        for i in range(len(self.data_loader.encode_pd.loc[0])):
            for j in range(len(self.data_loader.encode_pd.index)):
                if isinstance(self.data_loader.encode_pd.loc[j][i], list):
                    for k in self.data_loader.encode_pd.loc[j][i]:
                        sex_orientation = self.data_loader.category_dict.get(k[0], {}).get("sex", -1)
                        if sex_orientation != -1:
                            break
            if self.sex_orientation != -1:
                break

        for i in self.data_loader.props_dict:
            middle = int(len(self.data_loader.encode_pd.loc[0]) / 2)
            for j in range(middle):
                if "." in self.data_loader.props_dict[i][j] and self.data_loader.result_pd.loc[i][j] == -1:
                    count = 0
                    a = self.data_loader.encode_pd.at[i, j]
                    for k in a:
                        if 100000 not in k:
                            count += 1
                    if self.sex_orientation == 1:
                        self.man_wall_cell_count += count
                    else:
                        self.women_wall_cell_count += count
            for j in range(middle, len(self.data_loader.encode_pd.loc[0])):
                if "." in self.data_loader.props_dict[i][j] and self.data_loader.result_pd.loc[i][j] == -1:
                    count = 0
                    a = self.data_loader.encode_pd.at[i, j]
                    for k in a:
                        if 100000 not in k:
                            count += 1
                    if self.sex_orientation == 1:
                        self.women_wall_cell_count += count
                    else:
                        self.man_wall_cell_count += count

        self.shop_property_dict[self.Man_wall_upper_body_str] = 0
        self.shop_property_dict[self.Man_island_upper_body_str] = 0
        self.shop_property_dict[self.Woman_wall_upper_body_str] = 0
        self.shop_property_dict[self.Woman_island_upper_body_str] = 0
        self.shop_property_dict[self.Man_wall_pants_str] = 0
        self.shop_property_dict[self.Man_island_pants_str] = 0
        self.shop_property_dict[self.Women_wall_pants_str] = 0
        self.shop_property_dict[self.Women_island_pants_str] = 0
        for i in self.data_loader.bit_map:
            for j in self.data_loader.bit_map[i]:
                if self.data_loader.result_pd.loc[i][j] != -1:
                    continue
                # print(self.data_loader.bit_map[i][j]["category"])
                if i in self.data_loader.bit_map and j in self.data_loader.bit_map[i] and \
                        self.data_loader.bit_map[i][j]["sex"] == 1 and \
                        self.data_loader.bit_map[i][j]["category"][0] in self.Upper_body_category_list:
                    if "." in self.data_loader.props_dict[i][j]:
                        self.shop_property_dict[self.Man_wall_upper_body_str] += 1
                    else:
                        self.shop_property_dict[self.Man_island_upper_body_str] += 1
                elif i in self.data_loader.bit_map and j in self.data_loader.bit_map[i] and \
                        self.data_loader.bit_map[i][j]["sex"] == 1 and \
                        self.data_loader.bit_map[i][j]["category"][0] in self.Pants_category_list:
                    if "." in self.data_loader.props_dict[i][j]:
                        # print("get", self.data_loader.bit_map[i][j]["category"])
                        self.shop_property_dict[self.Man_wall_pants_str] += 1
                    else:
                        self.shop_property_dict[self.Man_island_pants_str] += 1
                elif i in self.data_loader.bit_map and j in self.data_loader.bit_map[i] and \
                        self.data_loader.bit_map[i][j]["sex"] == 1 and \
                        self.data_loader.bit_map[i][j]["category"][0] in self.Entire_body_list:
                    if "." in self.data_loader.props_dict[i][j]:
                        # print("get", self.data_loader.bit_map[i][j]["category"])
                        self.shop_property_dict[self.Man_wall_suite_str] += 1
                    else:
                        self.shop_property_dict[self.Man_island_suite_str] += 1
                elif i in self.data_loader.bit_map and j in self.data_loader.bit_map[i] and \
                        self.data_loader.bit_map[i][j]["sex"] == 0 and \
                        self.data_loader.bit_map[i][j]["category"][0] in self.Upper_body_category_list:
                    if "." in self.data_loader.props_dict[i][j]:
                        self.shop_property_dict[self.Woman_wall_upper_body_str] += 1
                    else:
                        self.shop_property_dict[self.Woman_island_upper_body_str] += 1
                elif i in self.data_loader.bit_map and j in self.data_loader.bit_map[i] and \
                        self.data_loader.bit_map[i][j]["sex"] == 0 and \
                        self.data_loader.bit_map[i][j]["category"][0] in self.Pants_category_list:
                    if "." in self.data_loader.props_dict[i][j]:
                        self.shop_property_dict[self.Women_wall_pants_str] += 1
                    else:
                        self.shop_property_dict[self.Women_island_pants_str] += 1
                elif i in self.data_loader.bit_map and j in self.data_loader.bit_map[i] and \
                        self.data_loader.bit_map[i][j]["sex"] == 0 and \
                        self.data_loader.bit_map[i][j]["category"][0] in self.Entire_body_list:
                    if "." in self.data_loader.props_dict[i][j]:
                        self.shop_property_dict[self.Women_wall_suite_str] += 1
                    else:
                        self.shop_property_dict[self.Women_island_suite_str] += 1

        # for i in self.shop_property_dict:
        #     print(i, self.shop_property_dict[i])

    def section_choose(self, position, result_pd):
        # print("position", position)
        index_cell_length = result_pd.shape[0] // INDEX_DIVIDE
        column_cell_length = result_pd.shape[1] // COLUMN_DIVIDE

        sex_orientation = -1
        for i in range(len(self.data_loader.encode_pd.loc[0])):
            for j in range(len(self.data_loader.encode_pd.index)):
                if isinstance(self.data_loader.encode_pd.loc[j][i], list):
                    for k in self.data_loader.encode_pd.loc[j][i]:
                        sex_orientation = self.data_loader.category_dict.get(k[0], {}).get("sex", -1)
                        if sex_orientation != -1:
                            break
            if sex_orientation != -1:
                break

        if sex_orientation == 1:
            return max(INDEX_DIVIDE - math.ceil((position[0] + 0.01) / index_cell_length), 0), \
                   min(math.ceil((position[1] + 0.01) / column_cell_length) - 1, COLUMN_DIVIDE - 1)
        else:
            return max(INDEX_DIVIDE - math.ceil((position[0] + 0.01) / index_cell_length), 0), \
                   max(COLUMN_DIVIDE - math.ceil((position[1] + 0.01) / column_cell_length), 0)

    # MODEL NEED
    def internal_sort(self, clothing, context, index, column):

        def get_item_score(a):
            score = 0
            if a["count"] - a["already_allocation_info"][0] > 0:
                score += 10**15
            score += 10 ** 14 if self.data_loader.bit_map[index][column]["sex"] == a[DataLoader.Sex_str] else 0

            if "." in self.data_loader.props_dict[index][column] and "墙面" == a[DataLoader.Position_str][0]:
                score += 10**13

            if "." in self.data_loader.props_dict[index][column] and len(a[DataLoader.Position_str]) > 1 and\
                    "墙面" == a[DataLoader.Position_str][1]:
                score += 10**12
            elif "." not in self.data_loader.props_dict[index][column] and "墙面" != a[DataLoader.Position_str][0]:
                score += 10 ** 12

            score += sum([a["category_score_list"][i] * (10 ** 9 / (1000 ** i)) for i in
                          range(len(a["category_score_list"]))])

            score -= 20 ** math.log(max((datetime.now() - a[DataLoader.Execute_time_str]).days, 1))

            return score

        clothing.sort(key=lambda a: get_item_score(a), reverse=True)

        return clothing

    def choose_best(self, section, clothing_list, context, index, column, rank=0):
        for i in clothing_list:
            i["category_score_list"] = [self.category_sort_info.get(j, {}).get(section[0], {}).get(section[1], 0) for j in
                                        i[DataLoader.Category_str]]

        clothing = self.internal_sort(clothing_list, context, index, column)

        if "." in self.data_loader.props_dict[index][column]:
            # print("clothing0", clothing[0]["name"], clothing[0][DataLoader.Position_str],
            #       clothing[0]["count"], clothing[0]["already_allocation_info"][0])
            pass

        if rank >= len(clothing):
            best_one = clothing[-1]
        else:
            if rank > 0 and clothing[rank]["count"] - clothing[rank]["already_allocation_info"][0] > 0:
                best_one = clothing[rank]
            else:
                best_one = clothing[0]

        if best_one["count"] - best_one["already_allocation_info"][0] > 0:
            best_one["already_allocation_info"][0] += 1

        clothing_name = best_one["name"]

        clothing_name = clothing_name.split("*")[0]
        clothing_name = clothing_name.split("￥")[0]

        return clothing_name, best_one

    def get_layout_clothing(self, plan):
        # print("before plan", plan)
        layout_clothing_dict = dict()
        star_clothing_dict = dict()
        display_clothing = copy.deepcopy(self.data_loader.display_clothing_dict)
        for i in display_clothing:
            if i.find("*星墙") != -1:
                star_clothing_dict[i] = display_clothing[i]
        for i in star_clothing_dict:
            display_clothing.pop(i)

        display_clothing_list = list()
        for i in display_clothing:

            display_clothing[i]["name"] = i

            root_category = display_clothing[i][DataLoader.Category_str][0]
            sex = display_clothing[i][DataLoader.Sex_str]
            wall = 0
            if "墙面" == display_clothing[i][DataLoader.Position_str][0]:
                wall = 2
            elif len(display_clothing[i][DataLoader.Position_str]) > 1 and \
                    "墙面" == display_clothing[i][DataLoader.Position_str][1]:
                wall = 1
            # island = 1 if "墙面" not in display_clothing[i][DataLoader.Position_str] else 0
            island = 1 if "中岛" in display_clothing[i][DataLoader.Position_str] else 0
            for clothing_property in [self.Man_wall_upper_body_str, self.Man_island_upper_body_str,
                                      self.Man_wall_pants_str, self.Man_island_pants_str,
                                      self.Woman_wall_upper_body_str, self.Woman_island_upper_body_str,
                                      self.Women_wall_pants_str, self.Women_island_pants_str]:
                display_clothing[i][clothing_property] = 0

            if root_category in self.Upper_body_category_list:
                if sex == 1:
                    display_clothing[i][self.Man_wall_upper_body_str] = wall
                    display_clothing[i][self.Man_island_upper_body_str] = island
                elif sex == 0:
                    display_clothing[i][self.Woman_wall_upper_body_str] = wall
                    display_clothing[i][self.Woman_island_upper_body_str] = island

            if root_category in self.Pants_category_list:
                if sex == 1:
                    display_clothing[i][self.Man_wall_pants_str] = wall
                    display_clothing[i][self.Man_island_pants_str] = island
                elif sex == 0:
                    display_clothing[i][self.Women_wall_pants_str] = wall
                    display_clothing[i][self.Women_island_pants_str] = island

            display_clothing_list.append(display_clothing[i])

        remove_star = set()
        for index in range(len(self.data_loader.props_dict)):
            row = len(self.data_loader.props_dict) - 1 - index
            for j in self.data_loader.props_dict[row]:
                if "*" in self.data_loader.props_dict[row][j] and "星中" in self.data_loader.props_dict[row][j]["*"]:
                    for display in display_clothing_list:
                        name = display["name"]
                        if name.find("星中") == -1:
                            continue
                        # print("*name", name)

                        sex = "男" if display[DataLoader.Sex_str] == 1 else "女"
                        props_full_name = name.split("*")[-1]
                        if sex not in props_full_name:
                            props_full_name = sex + props_full_name
                        if props_full_name == self.data_loader.props_dict[row][j]["*"]:
                            remove_star.add(display["name"])
                            if "current_cell_num" not in display:
                                display["current_cell_num"] = 0
                            if ":" in self.data_loader.props_dict[row][j]:
                                self.data_loader.result_pd.at[row, j] = "/".join([name.split("*")[0].split(
                                    ":")[0] for k in range(len(
                                    self.data_loader.encode_pd.at[row, j]))]) + ":" + self.data_loader.props_dict[
                                                                            row][j][":"] + "*" + props_full_name
                            else:
                                self.data_loader.result_pd.at[row, j] = "/".join([name.split("*")[0].split(
                                    ":")[0] for k in range(len(
                                    self.data_loader.encode_pd.at[row, j]))]) + "*" + props_full_name
                            display["current_cell_num"] += 1
                            if "already_allocation_num" not in display:
                                display["already_allocation_info"] = [0, []]
                            display["already_allocation_info"][0] = display["current_cell_num"]
                            display["already_allocation_info"][1].append([row, j])
                            display["count"] = display["current_cell_num"]

        for i in remove_star:
            remove_item = None
            for item in display_clothing_list:
                if item["name"] == i:
                    for clothing_property in [self.Man_island_upper_body_str, self.Man_island_pants_str,
                                              self.Woman_island_upper_body_str, self.Women_island_pants_str]:
                        if item[clothing_property] > 0:
                            plan[clothing_property] -= item["count"]
                    remove_item = item
                    break
            if remove_item is not None:
                display_clothing_list.remove(remove_item)

        for i in display_clothing_list:
            if "threshold_value" not in i:
                i["threshold_value"] = 0
            if "count" not in i:
                i["count"] = 0

        while plan[self.Man_wall_upper_body_str] + plan[self.Man_wall_pants_str] + \
                plan[self.Woman_wall_upper_body_str] + plan[self.Women_wall_pants_str] > 0:

            man_wall_pants_less = True if len(list(filter(lambda a: a[self.Man_wall_pants_str] == 2 and
                                                              a["count"] < 2, display_clothing_list))) > 0 else False
            man_wall_upper_less = True if len(list(filter(lambda a: a[self.Man_wall_upper_body_str] == 2 and
                                                                    a["count"] < 2, display_clothing_list))) > 0 else False

            woman_wall_pants_less = True if len(list(filter(lambda a: a[self.Women_wall_pants_str] == 2 and
                                                                a["count"] < 2, display_clothing_list))) > 0 else False

            woman_wall_upper_less = True if len(list(filter(lambda a: a[self.Woman_wall_upper_body_str] == 2 and
                                                                a["count"] < 2, display_clothing_list))) > 0 else False

            man_wall_pants_less_level2 = True if len(list(filter(lambda a: a[self.Man_wall_pants_str] == 1 and
                                                              a["count"] == 0, display_clothing_list))) > 0 else False
            man_wall_upper_less_level2 = True if len(list(filter(lambda a: a[self.Man_wall_upper_body_str] == 1 and
                                                               a["count"] == 0, display_clothing_list))) > 0 else False

            woman_wall_pants_less_level2 = True if len(list(filter(lambda a: a[self.Women_wall_pants_str] == 1 and
                                                                a["count"] == 0, display_clothing_list))) > 0 else False
            woman_wall_upper_less_level2 = True if len(list(filter(lambda a: a[self.Woman_wall_upper_body_str] == 1 and
                                                                 a["count"] == 0, display_clothing_list))) > 0 else False

            # 分配墙面指标
            ready = False
            display_clothing_list.sort(key=lambda a: (5-a[DataLoader.Importance_str])*100 + 10**10 if 4*a[
                "count"] - sum(a[DataLoader.Style_count_str]) < 0 else sum(a[DataLoader.Style_count_str])-4*a["count"])
            for i in display_clothing_list:

                can_use = False
                if man_wall_pants_less and i["count"] < 2 and i[self.Man_wall_pants_str] == 2 and \
                        plan[self.Man_wall_pants_str] > 0:
                    can_use = True
                elif man_wall_upper_less and i["count"] < 2 and i[self.Man_wall_upper_body_str] == 2 and \
                        plan[self.Man_wall_upper_body_str] > 0:
                    can_use = True
                elif woman_wall_pants_less and i["count"] < 2 and i[self.Women_wall_pants_str] == 2 and \
                        plan[self.Women_wall_pants_str] > 0:
                    can_use = True
                elif woman_wall_upper_less and i["count"] < 2 and i[self.Woman_wall_upper_body_str] == 2 and \
                        plan[self.Woman_wall_upper_body_str] > 0:
                    can_use = True
                if can_use is True:
                    i["threshold_value"] = 1
                    i["sort_important"] = i[DataLoader.Importance_str]
                    ready = True
                    break
            # 分配 墙面 和 小岛 指标
            if ready is False:
                for i in display_clothing_list:
                    can_use = False
                    if man_wall_pants_less_level2 and i["count"] < 2 and i[self.Man_wall_pants_str] == 1 and plan[self.Man_wall_pants_str] > 0:
                        can_use = True
                    elif man_wall_upper_less_level2 and i["count"] < 2 and i[self.Man_wall_upper_body_str] == 1 and plan[self.Man_wall_upper_body_str] > 0:
                        can_use = True
                    elif woman_wall_pants_less_level2 and i["count"] < 2 and i[self.Women_wall_pants_str] == 1 and plan[self.Women_wall_pants_str] > 0:
                        can_use = True
                    elif woman_wall_upper_less_level2 and i["count"] < 2 and i[self.Woman_wall_upper_body_str] == 1 and plan[self.Woman_wall_upper_body_str] > 0:
                        can_use = True
                    if can_use is True:
                        i["threshold_value"] = 1
                        i["sort_important"] = i[DataLoader.Importance_str]
                        ready = True
                        break
            # 满配额，在分配 指标
            if ready is False:
                for i in display_clothing_list:
                    can_use = False
                    count = 0
                    if i[self.Women_wall_pants_str] > 0 and plan[self.Women_wall_pants_str] > 0:
                        can_use = True
                        if i["count"] == 2 and plan[self.Women_wall_pants_str] >= 2:
                            count = 2
                        else:
                            count = 1
                    elif i[self.Woman_wall_upper_body_str] > 0 and plan[self.Woman_wall_upper_body_str] > 0:
                        can_use = True
                        if i["count"] == 2 and plan[self.Women_wall_pants_str] >= 2:
                            count = 2
                        else:
                            count = 1
                    elif i[self.Man_wall_pants_str] > 0 and plan[self.Man_wall_pants_str] > 0:
                        can_use = True
                        if i["count"] == 2 and plan[self.Man_wall_pants_str] >= 2:
                            count = 2
                        else:
                            count = 1
                    elif i[self.Man_wall_upper_body_str] > 0 and plan[self.Man_wall_upper_body_str] > 0:
                        can_use = True
                        if i["count"] == 2 and plan[self.Man_wall_upper_body_str] >= 2:
                            count = 2
                        else:
                            count = 1
                    if can_use is True:
                        i["threshold_value"] = count
                        i["sort_important"] = i[DataLoader.Importance_str]
                        ready = True
                        break

            assert ready is True

            # 具体指标分配
            for clothing_property in [self.Man_wall_upper_body_str, self.Man_wall_pants_str,
                                      self.Woman_wall_upper_body_str, self.Women_wall_pants_str]:

                while plan[clothing_property] > 0:
                    # print(clothing_property, plan[clothing_property])

                    display_clothing_list.sort(
                        key=lambda a: a[clothing_property] * (10 ** 11) + min(a.get("threshold_value", 0), 1) * (
                                10 ** 10) + (5 - a.get("sort_important", 5)) * (10 ** 9), reverse=True)
                    a = None
                    for i in range(len(display_clothing_list)):
                        if (display_clothing_list[i][DataLoader.Style_count_str][0] > 0 or
                            display_clothing_list[i][DataLoader.Style_count_str][1] > 0) and \
                                display_clothing_list[i][clothing_property] > 0 and \
                                display_clothing_list[i]["threshold_value"] > 0:
                            a = display_clothing_list[i]

                            a["sort_important"] += 1
                            # print("recurrent", a["name"], a["count"], a["threshold_value"])
                            a["count"] += 1
                            plan[clothing_property] -= 1
                            a["threshold_value"] -= 1
                    if a is None:
                        break

        # for i in display_clothing_list:
        #     if i[self.Man_wall_upper_body_str] > 0 or i[self.Man_wall_pants_str] > 0 or i[self.Woman_wall_upper_body_str] > 0 and i[self.Women_wall_pants_str] > 0:
        #         print("what a fuck", i["name"], i["count"])

        display_clothing_list.sort(key=lambda a: (5 - a[DataLoader.Importance_str])*1000+sum(a[DataLoader.Style_count_str]), reverse=True)
        for i in display_clothing_list:
            for clothing_property in [self.Man_island_upper_body_str, self.Man_island_pants_str,
                                      self.Woman_island_upper_body_str, self.Women_island_pants_str]:
                if i[clothing_property] > 0 and plan[clothing_property] > 0 and i["count"] == 0:
                    plan[clothing_property] -= 1
                    i["count"] = 1

        # 中岛 服装分配
        first_circle = True
        while sum(plan.values()) > 0:
            # print("fuck!!!!!fuck!!!!!fuck!!!!!", sum(plan.values()), plan[self.Man_wall_upper_body_str],
            #       plan[self.Man_island_upper_body_str], plan[self.Man_wall_pants_str], plan[self.Man_island_pants_str],
            #       plan[self.Woman_wall_upper_body_str], plan[self.Woman_island_upper_body_str],
            #       plan[self.Women_wall_pants_str], plan[self.Women_island_pants_str])

            for i in display_clothing_list:
                i["sort_important"] = i[DataLoader.Importance_str]
                if i[self.Women_wall_pants_str] == 2 or i[self.Woman_wall_upper_body_str] == 2 or \
                        i[self.Man_wall_pants_str] == 2 or i[self.Man_wall_upper_body_str] == 2:
                    continue
                base_count = int((i[DataLoader.Style_count_str][0] + i[DataLoader.Style_count_str][1]) / 2)
                i["threshold_value"] = min(math.ceil(self.Import_rate[i[DataLoader.Importance_str]] * base_count), 3)

                if i["count"] + i["threshold_value"] > 6 and i["threshold_value"] > 0:
                    i["threshold_value"] = 1

                if i[DataLoader.Importance_str] <= 2 and first_circle is True:
                    if i["name"].find("￥") != -1:
                        second_props_num = self.data_loader.second_props_num_dict.get(i["name"].split("￥")[-1], 0)
                        if second_props_num > i["threshold_value"]:
                            i["threshold_value"] = min(second_props_num, base_count)

            first_circle = False

            for clothing_property in [self.Man_island_upper_body_str, self.Man_island_pants_str,
                                      self.Woman_island_upper_body_str, self.Women_island_pants_str]:

                while plan[clothing_property] > 0:

                    display_clothing_list.sort(
                        key=lambda a: a[clothing_property] * (10 ** 11) + min(a["threshold_value"], 1) * (
                                10 ** 10) + (5 - a["sort_important"]) * (10 ** 9), reverse=True)
                    a = None
                    for i in range(len(display_clothing_list)):
                        if (display_clothing_list[i][DataLoader.Style_count_str][0] > 0 or
                            display_clothing_list[i][DataLoader.Style_count_str][1] > 0) and \
                                display_clothing_list[i][clothing_property] > 0 and \
                                display_clothing_list[i]["threshold_value"] > 0:
                            a = display_clothing_list[i]
                            if a[DataLoader.Style_count_str][0] > 0:
                                a[DataLoader.Style_count_str][0] -= 1
                                a[DataLoader.Style_count_str][1] -= 1
                            elif a[DataLoader.Style_count_str][1] > 0:
                                a[DataLoader.Style_count_str][1] -= 1
                            if a[DataLoader.Style_count_str][0] <= 0 and a[DataLoader.Style_count_str][1] < 5:
                                a["sort_important"] += 1

                            a["count"] += 1
                            plan[clothing_property] -= 1
                            a["threshold_value"] -= 1
                            break
                    if a is None:
                        break

        # 优先分配，特殊制定位置商品
        mean_cell_style_count = 1
        for index in range(len(self.data_loader.props_dict)):
            row = len(self.data_loader.props_dict) - 1 - index
            for j in self.data_loader.props_dict[row]:
                if "￥" in self.data_loader.props_dict[row][j]:
                    for display in display_clothing_list:
                        name = display["name"]
                        count = display["count"]
                        # print("￥name", name)
                        props_full_name = name.split("￥")[-1]
                        if props_full_name == self.data_loader.props_dict[row][j]["￥"]:
                            if "current_cell_num" not in display:
                                display["current_cell_num"] = 0
                            if display["current_cell_num"]*mean_cell_style_count < count:
                                if ":" in self.data_loader.props_dict[row][j]:
                                    self.data_loader.result_pd.at[row, j] = "/".join([name.split("￥")[0] for k in range(
                                        len(self.data_loader.encode_pd.at[
                                                row, j]))]) + ":" + self.data_loader.props_dict[
                                        row][j][":"] + "￥" + props_full_name
                                else:
                                    self.data_loader.result_pd.at[row, j] = "/".join([name.split("￥")[0] for k in range(
                                        len(self.data_loader.encode_pd.at[row, j]))]) + "￥" + props_full_name
                                display["current_cell_num"] += 1
                                if "already_allocation_num" not in display:
                                    display["already_allocation_info"] = [0, []]
                                display["already_allocation_info"][0] = display["current_cell_num"]
                                display["already_allocation_info"][1].append([row, j])
                                # print("current_cell_num", display)

        # print(self.data_loader.result_pd)
        # 合并没有分到位置的服装
        self.merge_residue(display_clothing_list)

        for i in display_clothing_list:
            # if i["count"] > 1:
            if i["count"] == 0:
                print("what a fuck@@@@@", i["name"], i["count"])

        for i in display_clothing_list:
            layout_clothing_dict[i["name"]] = i
        return layout_clothing_dict, star_clothing_dict

    @classmethod
    def merge_residue(cls, values):

        def sort_value(a, current_value):
            score = 0
            score += 10**9 if a[DataLoader.Sex_str] == current_value[DataLoader.Sex_str] else 0
            score += 10**8 if a[DataLoader.Importance_str] == current_value[DataLoader.Importance_str] else 0
            for i in range(min(len(a[DataLoader.Category_str]), len(current_value[DataLoader.Category_str]))):
                if a[DataLoader.Category_str][i] == current_value[DataLoader.Category_str][i]:
                    score += (10**8)/(10**i)
            return score

        not_set_values = list()
        for i in range(len(values)):
            if values[i]["count"] == 0:
                not_set_values.append(values[i])
        for value in not_set_values:
            values.remove(value)
        for value in not_set_values:
            values.sort(key=lambda a: sort_value(a, value), reverse=True)
            anchor_value = values[0]
            if len(anchor_value["name"].split("￥")) > 1:
                item = "￥"
                anchor_name = anchor_value["name"].split(item)[0]
                anchor_prop_name = anchor_value["name"].split(item)[1]

                anchor_value["name"] = anchor_name + "&" + value["name"] + item + anchor_prop_name
            elif len(anchor_value["name"].split("*")) > 1:
                item = "*"
                anchor_name = anchor_value["name"].split(item)[0]
                anchor_prop_name = anchor_value["name"].split(item)[1]
                anchor_value["name"] = anchor_name + "&" + value["name"] + item + anchor_prop_name
            else:
                anchor_name = anchor_value["name"].split(":")[0]
                anchor_value["name"] = anchor_name + "&" + value["name"]
            print("合并(%s--%s)-----%s" % (anchor_name, value["name"], anchor_value["name"]))

    # 分配制定星墙服装
    def layout_star_wall(self):
        for i in self.data_loader.props_dict:
            for j in self.data_loader.props_dict[i]:
                if "*" in self.data_loader.props_dict[i][j] and self.data_loader.props_dict[i][j]["*"].find("星墙") != -1:
                    # print("djdjdjdj", self.data_loader.props_dict[i][j]["*"])
                    for name in self.data_loader.display_clothing_dict:
                        # print(name, self.data_loader.display_clothing_dict[name][DataLoader.Sex_str] == 0)
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
        # print(self.data_loader.result_pd)

    def layout_suit_wall(self):
        man_suit_prop_dict = dict()
        woman_suit_prop_dict = dict()
        for i in self.data_loader.props_dict:
            for j in self.data_loader.props_dict[i]:
                if ":" in self.data_loader.props_dict[i][j] and \
                        self.data_loader.props_dict[i][j][":"].find("高支") != -1 and \
                        self.data_loader.result_pd.at[i, j] == -1:
                    prop = self.data_loader.props_dict[i][j][":"]
                    if self.data_loader.bit_map[i][j]["sex"] == 1:
                        if prop not in man_suit_prop_dict:
                            man_suit_prop_dict[prop] = list()
                        man_suit_prop_dict[prop].append([i, j])
                    else:
                        if prop not in woman_suit_prop_dict:
                            woman_suit_prop_dict[prop] = list()
                        woman_suit_prop_dict[prop].append([i, j])

        man_suit_prop_list = list(man_suit_prop_dict.items())
        man_suit_prop_list.sort(key=lambda a: int(a[0][-1]), reverse=True)

        woman_suit_prop_list = list(woman_suit_prop_dict.items())
        woman_suit_prop_list.sort(key=lambda a: int(a[0][-1]), reverse=True)

        man_suit_clothing_dict = dict()
        woman_suit_clothing_dict = dict()
        for name in self.data_loader.display_clothing_dict:
            if name.find("*") != -1 or name.find("￥") != -1:
                continue
            if self.data_loader.display_clothing_dict[name][DataLoader.Category_str][0] in self.Suit_list:
                if self.data_loader.display_clothing_dict[name][DataLoader.Sex_str] == 1:
                    man_suit_clothing_dict[name] = self.data_loader.display_clothing_dict[name]
                else:
                    woman_suit_clothing_dict[name] = self.data_loader.display_clothing_dict[name]

        man_suit_clothing_list = list(man_suit_clothing_dict.items())
        man_suit_clothing_list.sort(key=lambda a: (5-a[1][DataLoader.Importance_str]), reverse=True)

        woman_suit_clothing_list = list(woman_suit_clothing_dict.items())
        woman_suit_clothing_list.sort(key=lambda a: (5 - a[1][DataLoader.Importance_str]), reverse=True)

        for suit_clothing_list, suit_prop_list in zip([man_suit_clothing_list, woman_suit_clothing_list],
                                                      [man_suit_prop_list, woman_suit_prop_list]):
            if len(suit_clothing_list) == 0 or len(suit_prop_list) == 0:
                continue
            for suit in suit_clothing_list:
                # print(suit)
                can_remove = False
                count = min(len(suit_prop_list[0][1]), 2) if \
                    suit[1][DataLoader.Importance_str] in [4, 5] else \
                    min(len(suit_prop_list[0][1]), int(sum(suit[1][DataLoader.Style_count_str])/4))
                suit_prop = suit_prop_list[0]
                while count > 0 and len(suit_prop[1]) > 0:
                    can_remove = True
                    postion = suit_prop[1][0]
                    self.data_loader.result_pd.at[postion[0], postion[1]] = "%s:%s" % (suit[0], suit_prop[0])
                    suit_prop[1].remove(postion)
                    count -= 1
                if can_remove is True:
                    print(suit[0])
                    self.data_loader.display_clothing_dict.pop(suit[0])
        # print(self.data_loader.result_pd)

    def layout(self, csv_path, plan_path, context=None):
        print(csv_path)
        self.data_loader = DataLoader()
        self.data_loader.load_csv(csv_path, plan_path)

        self.layout_star_wall()
        self.layout_suit_wall()

        self.init()

        display_clothing_dict, star = self.get_layout_clothing(self.shop_property_dict)

        clothing_list = list()
        for key, value in display_clothing_dict.items():

            if "already_allocation_info" not in value:
                value["already_allocation_info"] = [0, []]
            clothing_list.append(value)

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
                        result_list = list()
                        category_list = list()
                        encode_list = self.data_loader.encode_pd.at[index, column]
                        rank = 0
                        score_list = list()
                        have_regu = False
                        for k in encode_list:
                            if isinstance(k, list):
                                if 100000 in k:
                                    have_regu = True
                                    break
                        if have_regu is True:
                            result_list.append("矩框")
                        result_name, result = self.choose_best(section, clothing_list, context, index, column, 0)
                        result_list.append(result_name)
                        score_list.append(result[DataLoader.Order_value_str])
                        category_list.append(result[DataLoader.Category_str])
                        result["rank"] = rank
                        rank += 1

                        result_name = "/".join(result_list)
                        if len(self.data_loader.props_dict[index][column]) > 0 and "." not in \
                                self.data_loader.props_dict[index][column]:
                            if ":" in self.data_loader.props_dict[index][column]:
                                result_name += "%s%s" % (":", self.data_loader.props_dict[index][column][":"])
                            if "*" in self.data_loader.props_dict[index][column]:
                                result_name += "%s%s" % ("*", self.data_loader.props_dict[index][column]["*"])
                            if "￥" in self.data_loader.props_dict[index][column]:
                                result_name += "%s%s" % ("￥", self.data_loader.props_dict[index][column]["￥"])

                        return [result_name, score_list, category_list]

                    self.data_loader.result_pd.at[index_length - j, fix_i] = get_result_list(index_length - j, fix_i)

                    small_search_fix_i_neighbor = fix_i - 1
                    while small_search_fix_i_neighbor > 0 and \
                            self.data_loader.result_pd.at[index_length - j, small_search_fix_i_neighbor] in [-1]:

                        if "￥" in self.data_loader.props_dict[index_length - j][fix_i] and \
                                "￥" in self.data_loader.props_dict[index_length - j][small_search_fix_i_neighbor] and \
                                self.data_loader.props_dict[index_length - j][fix_i]["￥"] == \
                                self.data_loader.props_dict[index_length - j][small_search_fix_i_neighbor]["￥"]:

                            self.data_loader.result_pd.at[index_length - j, small_search_fix_i_neighbor] = get_result_list(
                                index_length - j, small_search_fix_i_neighbor)
                            small_search_fix_i_neighbor -= 1
                        elif ":" in self.data_loader.props_dict[index_length - j][fix_i] and \
                                ":" in self.data_loader.props_dict[index_length - j][small_search_fix_i_neighbor] and \
                                self.data_loader.props_dict[index_length - j][fix_i][":"] == \
                                self.data_loader.props_dict[index_length - j][small_search_fix_i_neighbor][":"]:

                            self.data_loader.result_pd.at[index_length - j, small_search_fix_i_neighbor] = get_result_list(
                                index_length - j, small_search_fix_i_neighbor)
                            small_search_fix_i_neighbor -= 1
                        else:
                            break

                    big_search_fix_i_neighbor = fix_i + 1
                    while big_search_fix_i_neighbor < len(self.data_loader.result_pd.loc[0]) - 2 and \
                            self.data_loader.result_pd.at[index_length - j, big_search_fix_i_neighbor] in [-1]:

                        if "￥" in self.data_loader.props_dict[index_length - j][fix_i] and \
                                "￥" in self.data_loader.props_dict[index_length - j][big_search_fix_i_neighbor] and \
                                self.data_loader.props_dict[index_length - j][fix_i]["￥"] == \
                                self.data_loader.props_dict[index_length - j][big_search_fix_i_neighbor]["￥"]:

                            self.data_loader.result_pd.at[index_length - j, big_search_fix_i_neighbor] = get_result_list(
                                index_length - j, big_search_fix_i_neighbor)
                            big_search_fix_i_neighbor += 1
                        elif ":" in self.data_loader.props_dict[index_length - j][fix_i] and \
                                ":" in self.data_loader.props_dict[index_length - j][big_search_fix_i_neighbor] and \
                                self.data_loader.props_dict[index_length - j][fix_i][":"] == \
                                self.data_loader.props_dict[index_length - j][big_search_fix_i_neighbor][":"]:

                            self.data_loader.result_pd.at[index_length - j, big_search_fix_i_neighbor] = get_result_list(
                                index_length - j, big_search_fix_i_neighbor)
                            big_search_fix_i_neighbor += 1
                        else:
                            break

        self.resort()

        self.reassign_prop()

        self.result_pd_cell_to_str()

        self.data_loader.result_pd.to_csv(
            os.getcwd() + "/data/result/" + "%s.csv" % csv_path.split("/")[-1].split(".")[0])
        print("保存至[%s" % os.getcwd() + "/data/result/" + "%s.csv]" % csv_path.split("/")[-1].split(".")[0])

    def result_pd_cell_to_str(self):
        for i in range(len(self.data_loader.result_pd.index)):
            for j in range(len(self.data_loader.result_pd.loc[0])):
                if isinstance(self.data_loader.result_pd.loc[i][j], list):
                    self.data_loader.result_pd.at[i, j] = self.data_loader.result_pd.at[i, j][0]

    def island_sorted(self, sex):
        for i in range(0, len(self.data_loader.result_pd.loc[0])):
            sorted_list = list()
            a = self.data_loader.result_pd[i]
            for j in range(len(a)):
                if isinstance(a[j], list) and "." not in self.data_loader.props_dict[j][i]\
                        and self.data_loader.bit_map[j][i]["sex"] == sex:
                    sorted_list.append(a[j])
                    self.data_loader.result_pd.at[j, i] = [-1, a[j][0].split(":")[1]]
                    # a[j] = [-1, a[j][0].split(":")[1]]
            sorted_list.sort(key=lambda a: a[1][0])
            for item in sorted_list:
                for j in range(len(a)):
                    if isinstance(a[j], list) and a[j][0] == -1:
                        self.data_loader.result_pd.at[j, i] = [item[0].split(":")[0]+":"+a[j][1], item[1], item[2]]
                        # a[j] = [item[0].split(":")[0]+":"+a[j][1], item[1], item[2]]
                        break

    def modify_need_sort_wall(self, i, sorted_dict):
        a = self.data_loader.result_pd[i]
        for j in range(len(a)):
            if isinstance(a[j], list) and "." in self.data_loader.props_dict[j][i]:
                if str(a[j][0]) == "矩框":
                    continue
                clothing = a[j][0].split("/")[0]
                if clothing == "矩框":
                    clothing = a[j][0].split("/")[1]

                if clothing not in sorted_dict:
                    sorted_dict[clothing] = {
                        "count": 0, "score": a[j][1][0],
                        "category": a[j][2][0][0]}
                sorted_dict[clothing]["count"] += 1
                if len(a[j][0].split("/")) > 1:
                    self.data_loader.result_pd.at[j, i] = ["矩框", -1]
                    # a[j] = ["矩框", -1]
                else:
                    self.data_loader.result_pd.at[j, i] = [-1]
                    # a[j] = [-1]

    def modify_sort_wall(self, a, i, j, sorted_list):
        if isinstance(a[j], list) and "." in self.data_loader.props_dict[i][j]:
            item = sorted_list[0]

            if a[j][0] == -1:
                self.data_loader.result_pd.at[i, j] = item[0]
                # a[j] = item[0]
            else:
                self.data_loader.result_pd.at[i, j] = "/".join(["矩框", item[0]])
                # a[j] = "/".join(["矩框", item[0]])
            item[1]["count"] -= 1
            if item[1]["count"] < 1:
                sorted_list.remove(item)

    def resort(self):

        # men island sort
        self.island_sorted(1)
        # women island sort
        self.island_sorted(0)

        # left wall
        sorted_dict = dict()
        for i in range(0, int(len(self.data_loader.result_pd.loc[0])/2)):
            self.modify_need_sort_wall(i, sorted_dict)

        sorted_list = list(sorted_dict.items())
        sorted_list.sort(key=lambda a: a[1]["score"] + [10000 if "上衣" in a[1]["category"] else 0][0])

        # print("left wall sorted_list", sorted_list)

        for i in range(len(self.data_loader.result_pd.index)):
            a = self.data_loader.result_pd.loc[i]
            for j in range(int(len(self.data_loader.result_pd.loc[0])/2)-1, -1, -1):
                self.modify_sort_wall(a, i, j, sorted_list)
        # print("left wall sorted_list", sorted_list)

        # right wall
        sorted_dict = dict()
        for i in range(int(len(self.data_loader.result_pd.loc[0]) / 2), len(self.data_loader.result_pd.loc[0])):
            self.modify_need_sort_wall(i, sorted_dict)

        sorted_list = list(sorted_dict.items())
        sorted_list.sort(key=lambda a: a[1]["score"] + [10000 if "上衣" in a[1]["category"] else 0][0])

        # print("right wall sorted_list", sorted_list)

        for i in range(len(self.data_loader.result_pd.index)):
            a = self.data_loader.result_pd.loc[i]
            for j in range(int(len(self.data_loader.result_pd.loc[0])/2), len(self.data_loader.result_pd.loc[0])):
                self.modify_sort_wall(a, i, j, sorted_list)

        # print("right wall sorted_list", sorted_list)

    def reassign_prop(self):
        clothing_prop_dict = dict()
        prop_clothing_dict = dict()
        for i in range(len(self.data_loader.result_pd.index)):
            for j in range(len(self.data_loader.result_pd.loc[i])):
                if i in self.data_loader.props_dict and j in self.data_loader.props_dict[i] and \
                        ":" in self.data_loader.props_dict[i][j]:
                    if isinstance(self.data_loader.result_pd.loc[i][j], list):
                        clothing_name = str(self.data_loader.result_pd.loc[i][j][0]).split(":")[0]
                        props_name = self.data_loader.props_dict[i][j][":"]
                        if clothing_name not in clothing_prop_dict:
                            clothing_prop_dict[clothing_name] = dict()
                        if props_name not in clothing_prop_dict[clothing_name]:
                            clothing_prop_dict[clothing_name][props_name] = list()
                        clothing_prop_dict[clothing_name][props_name].append([i, j])
                        if props_name not in prop_clothing_dict:
                            prop_clothing_dict[props_name] = dict()
                        if clothing_name not in prop_clothing_dict[props_name]:
                            prop_clothing_dict[props_name][clothing_name] = list()
                        prop_clothing_dict[props_name][clothing_name].append([i, j])
        for clothing_name in clothing_prop_dict:
            if len(clothing_prop_dict[clothing_name]) > 0:
                prop = None
                max_count = None
                for key, value in clothing_prop_dict[clothing_name].items():
                    if prop is None:
                        prop = key
                        max_count = len(value)
                    elif max_count < len(value):
                        prop = key
                        max_count = len(value)
                for key, value in clothing_prop_dict[clothing_name].items():
                    if key == prop:
                        continue
                    for position in value:
                        for i in range(max(position[0]-3, 0), min(position[0]+3, len(self.data_loader.result_pd.loc[0]))):
                            for j in range(max(position[1] - 3, 0), min(position[1] + 3, len(self.data_loader.result_pd[0]))):
                                if i in self.data_loader.props_dict and j in self.data_loader.props_dict[i] and ":" in self.data_loader.props_dict[i][j]:
                                    if self.data_loader.props_dict[i][j][":"] != self.data_loader.props_dict[position[0]][position[1]][":"]:
                                        continue
                                    name = None
                                    if isinstance(self.data_loader.result_pd.loc[i][j], list):
                                        name = self.data_loader.result_pd.loc[i][j][0]
                                    elif str(self.data_loader.result_pd.loc[i][j]).find(":") != -1:
                                        name = str(self.data_loader.result_pd.loc[i][j])
                                    if name is not None:
                                        self.data_loader.result_pd.at[(position[0], position[1])] = name


if __name__ == "__main__":
    shop_property_csv_path = "/Users/quantum/code/StoreLayout/data/details/20181024/shop_property_utf8.csv"
    root_path = "/Users/quantum/code/StoreLayout/data/details/20181024/utf8_files/"
    plan_path = "/Users/quantum/code/StoreLayout/data/plan/2018_15_plan_utf8.csv"
    layout = LayoutManager()
    print(os.listdir(root_path))
    # exit()
    # ['西充.csv', '仁寿.csv', '高县.csv', '泸州.csv', '阆中.csv', '富顺.csv', '大英.csv', '雅安.csv', '绵阳.csv', '资阳.csv']:
    for file_path in ['西充.csv', '仁寿.csv', '高县.csv', '泸州.csv', '阆中.csv', '富顺.csv', '大英.csv', '雅安.csv', '绵阳.csv', '资阳.csv']:
        layout.layout(os.path.join(root_path, file_path), plan_path, CONTEXT_DICT)
