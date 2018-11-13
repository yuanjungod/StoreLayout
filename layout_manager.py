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

    Import_rate = {
        1: 1,
        2: 3 / 5,
        3: 2 / 5,
        4: 1 / 4,
        5: 1 / 4
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
        # shop_pd = pd.read_csv(shop_property_path)
        # for i in shop_pd.index:
        #     shop_property = shop_pd.loc[i]
        #     self.shop_property_dict[shop_property[0]] = dict()
        #     self.shop_property_dict[shop_property[0]][self.Man_upper_body_str] = shop_property[3]
        #     self.shop_property_dict[shop_property[0]][self.Woman_upper_body_str] = shop_property[5]
        #     self.shop_property_dict[shop_property[0]][self.Man_pants_str] = shop_property[4]
        #     self.shop_property_dict[shop_property[0]][self.Women_pants_str] = shop_property[6]

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

        for i in self.shop_property_dict:
            print(i, self.shop_property_dict[i])

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
            # elif "." not in self.data_loader.props_dict[index][column] and "墙面" == a[DataLoader.Position_str][0]:
            #     score -= 10**12
            # elif "." in self.data_loader.props_dict[index][column] and "墙面" not in a[DataLoader.Position_str]:
            #     score -= 10 ** 12
            elif "." not in self.data_loader.props_dict[index][column] and "墙面" != a[DataLoader.Position_str][0]:
                score += 10 ** 12

            # score += 10**11 if self.data_loader.bit_map[index][column]["category"][0] == a[DataLoader.Category_str][0] else 0

            score += sum([a["category_score_list"][i] * (10 ** 9 / (1000 ** i)) for i in range(len(a["category_score_list"]))])

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
            print("clothing0", clothing[0]["name"], clothing[0][DataLoader.Position_str],
                  clothing[0]["count"], clothing[0]["already_allocation_info"][0])

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

    def get_layout_clothing(self, plan, total_cell_count):
        # print("before plan", plan)
        layout_clothing_dict = dict()
        star_clothing_dict = dict()
        display_clothing = copy.deepcopy(self.data_loader.display_clothing_dict)
        for i in display_clothing:
            if i.find("*星墙") != -1:
                star_clothing_dict[i] = display_clothing[i]
        for i in star_clothing_dict:
            display_clothing.pop(i)

        # for key, value in star_clothing_dict.items():
        #     star = value[DataLoader.Position_str][0]
        #     have_star_props = False
        #     for i in self.data_loader.props_dict:
        #         for j in self.data_loader.props_dict[i]:
        #             if "*" not in self.data_loader.props_dict[i][j]:
        #                 continue
        #             # print(star, self.data_loader.props_dict[i][j]["*"])
        #             if value[DataLoader.Sex_str] == 1 and "男" not in star:
        #                 star = "男" + star
        #             elif value[DataLoader.Sex_str] == 0 and "女" not in star:
        #                 star = "女" + star
        #             # print(star, self.data_loader.props_dict[i][j]["*"])
        #             if star == self.data_loader.props_dict[i][j]["*"]:
        #                 # print("fu")
        #                 have_star_props = True
        #
        #     if have_star_props:
        #         if value[DataLoader.Category_str][0] in self.Pants_category_list and value[DataLoader.Sex_str] == 1:
        #             # print("fu11", value[DataLoader.Style_str])
        #             plan[self.Man_wall_pants_str] -= value[DataLoader.Style_count_str][0]
        #         elif value[DataLoader.Category_str][0] in self.Pants_category_list and value[DataLoader.Sex_str] == 0:
        #             # print("fu12", value[DataLoader.Style_str])
        #             plan[self.Women_wall_pants_str] -= value[DataLoader.Style_count_str][0]
        #         elif value[DataLoader.Category_str][0] in self.Upper_body_category_list and value[DataLoader.Sex_str] == 0:
        #             # print("fu13", value[DataLoader.Style_str])
        #             plan[self.Woman_wall_upper_body_str] -= value[DataLoader.Style_count_str][0]
        #         elif value[DataLoader.Category_str][0] in self.Upper_body_category_list and value[DataLoader.Sex_str] == 1:
        #             # print("fu14", value[DataLoader.Style_str])
        #             plan[self.Man_wall_upper_body_str] -= value[DataLoader.Style_count_str][0]

        # print("after plan", plan)

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
            island = 1 if "墙面" not in display_clothing[i][DataLoader.Position_str] else 0
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

            # print("easy!", list(filter(lambda a: (a[self.Man_wall_pants_str] == 2 or
            #                                                    a[self.Man_wall_upper_body_str] == 2) and
            #                                                   a["count"] < 2, display_clothing_list)))

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

            # for i in display_clothing_list:
            #     if i["count"] > 1:
            #         print("what!!!", i["name"], i["count"], ready)
            ready = False
            display_clothing_list.sort(key=lambda a: (5-a[DataLoader.Importance_str])*100 + 10**10 if 4*a[
                "count"] - sum(a[DataLoader.Style_count_str]) < 0 else sum(a[DataLoader.Style_count_str])-4*a["count"])
            print("display_clothing_list", display_clothing_list[0]["name"], display_clothing_list[0][DataLoader.Style_count_str], display_clothing_list[0]["count"])
            for i in display_clothing_list:

                can_use = False
                if man_wall_pants_less and i["count"] < 2 and i[self.Man_wall_pants_str] == 2 and plan[self.Man_wall_pants_str] > 0:
                    can_use = True
                elif man_wall_upper_less and i["count"] < 2 and i[self.Man_wall_upper_body_str] == 2 and plan[self.Man_wall_upper_body_str] > 0:
                    can_use = True
                elif woman_wall_pants_less and i["count"] < 2 and i[self.Women_wall_pants_str] == 2 and plan[self.Women_wall_pants_str] > 0:
                    can_use = True
                elif woman_wall_upper_less and i["count"] < 2 and i[self.Woman_wall_upper_body_str] == 2 and plan[self.Woman_wall_upper_body_str] > 0:
                    can_use = True
                if can_use is True:
                    i["threshold_value"] = 1
                    i["sort_important"] = i[DataLoader.Importance_str]
                    ready = True
                    break
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
                            print("recurrent", a["name"], a["count"], a["threshold_value"])
                            a["count"] += 1
                            plan[clothing_property] -= 1
                            a["threshold_value"] -= 1
                    if a is None:
                        break

        for i in display_clothing_list:
            if i[self.Man_wall_upper_body_str] > 0 or i[self.Man_wall_pants_str] > 0 or i[self.Woman_wall_upper_body_str] > 0 and i[self.Women_wall_pants_str] > 0:
                print("what a fuck", i["name"], i["count"])

        for i in display_clothing_list:
            for clothing_property in [self.Man_island_upper_body_str, self.Man_island_pants_str,
                                      self.Woman_island_upper_body_str, self.Women_island_pants_str]:
                if i[clothing_property] > 0 and plan[clothing_property] > 0 and i["count"] == 0:
                    plan[clothing_property] -= 1
                    i["count"] = 1

        first_circle = True
        while sum(plan.values()) > 0:
            print("fuck!!!!!fuck!!!!!fuck!!!!!", sum(plan.values()), plan[self.Man_wall_upper_body_str],
                  plan[self.Man_island_upper_body_str], plan[self.Man_wall_pants_str], plan[self.Man_island_pants_str],
                  plan[self.Woman_wall_upper_body_str], plan[self.Woman_island_upper_body_str],
                  plan[self.Women_wall_pants_str], plan[self.Women_island_pants_str])

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
        for i in display_clothing_list:
            if i["count"] > 1:
                print("what a fuck@@@@@", i["name"], i["count"])

        for i in display_clothing_list:
            layout_clothing_dict[i["name"]] = i
        return layout_clothing_dict, star_clothing_dict

    def allocation_cell_count(self, layout_clothing_dict):
        values = list(layout_clothing_dict.values())
        for value in values:
            if "current_cell_num" not in value:
                value["current_cell_num"] = 0
            if "already_allocation_info" not in value:
                value["already_allocation_info"] = [0, []]
        total_cell = self.calculate() + sum([value["current_cell_num"] for value in values])

        # mean_cell_style_count = sum([value["count"] for key, value in layout_clothing_dict.items()]) / total_cell
        mean_cell_style_count = 1
        remain_cell = total_cell - sum([value["current_cell_num"] for value in values])

        def sort_wall_value(a, sex):
            value = 0
            if a[DataLoader.Sex_str] == sex and "墙面" == a[DataLoader.Position_str][0]:
                # print("sort_wall_value")
                value += 10**8
            elif len(a[DataLoader.Position_str]) > 1 and a[DataLoader.Sex_str] == sex and "墙面" == a[DataLoader.Position_str][1]:
                value += 10**7
            value += (a["count"] - a["current_cell_num"]*mean_cell_style_count)*100 + a[DataLoader.Importance_str]
            return value

        def normal_sort_value(a):
            value = 0
            if "墙面" == a[DataLoader.Position_str][0]:
                value -= 10**8
            value += (a["count"] - a["current_cell_num"] * mean_cell_style_count) * 100 + a[DataLoader.Importance_str]
            return value

        women_wall_current_cell_count = 0
        men_wall_current_cell_count = 0
        while remain_cell > 0:
            if women_wall_current_cell_count <= self.women_wall_cell_count+1:

                values.sort(key=lambda a: sort_wall_value(a, 0), reverse=True)
                values[0]["current_cell_num"] += 1
                remain_cell -= 1
                women_wall_current_cell_count += 1
                # print("women_wall_current_cell_count", values[0]["name"], values[0][DataLoader.Position_str],
                # values[0]["current_cell_num"])

            elif men_wall_current_cell_count <= self.man_wall_cell_count+1:

                values.sort(key=lambda a: sort_wall_value(a, 1), reverse=True)
                values[0]["current_cell_num"] += 1
                remain_cell -= 1
                men_wall_current_cell_count += 1
                # print("men_wall_current_cell_count", values[0]["name"], values[0][DataLoader.Position_str],
                # values[0]["current_cell_num"])
            else:
                values.sort(key=lambda a: normal_sort_value(a), reverse=True)
                values[0]["current_cell_num"] += 1
                remain_cell -= 1

        print("women_wall_current_cell_count", women_wall_current_cell_count, self.women_wall_cell_count)
        print("men_wall_current_cell_count", men_wall_current_cell_count, self.man_wall_cell_count)

        # self.merge_residue(values, mean_cell_style_count)

        for value in values:
            # print("after", value["name"], value["count"], value["current_cell_num"])
            # if "墙面" in value[DataLoader.Position_str] or "星墙1" in value[DataLoader.Position_str] or \
            #         "星墙2" in value[DataLoader.Position_str] or "星墙3" in value[DataLoader.Position_str]:
            #     print(value["name"], value[DataLoader.Position_str], value["current_cell_num"])
            if value["current_cell_num"] == 0:
                print("not be set", value["name"], value["count"])

    @classmethod
    def merge_residue(cls, values, mean_cell_style_count):

        def sort_value(a, current_value):
            score = 0
            score += 10**9 if a[DataLoader.Sex_str] == current_value[DataLoader.Sex_str] else 0
            for i in range(min(len(a[DataLoader.Category_str]), len(current_value[DataLoader.Category_str]))):
                if a[DataLoader.Category_str][i] == current_value[DataLoader.Category_str][i]:
                    score += (10**8)/(10**i)
            score += (a["count"] - a["current_cell_num"]*mean_cell_style_count)*100
            return score

        not_set_values = list()
        for i in range(len(values)):
            if values[i]["current_cell_num"] == 0:
                not_set_values.append(values[i])
        for value in not_set_values:
            values.remove(value)
        for value in not_set_values:
            values.sort(key=lambda a: sort_value(a, value), reverse=True)

            for i in range(len(values)):
                anchor_value = values[i]
                if anchor_value["current_cell_num"] >= 2:

                    anchor_value["current_cell_num"] -= 1
                    value["current_cell_num"] = 1
                    if anchor_value["current_cell_num"] == 2 and len(value["name"].split("&")) == 1:
                        value["current_cell_num"] = 1
                        value["name"] = values[0]["name"].split("*")[0].split("￥")[0] + "&" + value["name"]
                        value[DataLoader.Category_str] = anchor_value[DataLoader.Category_str]
                    break
        for value in not_set_values:
            values.append(value)

    def layout_star(self):
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

    def calculate(self):
        total_cell = 0
        for i in self.data_loader.result_pd.index:
            for j in range(len(self.data_loader.result_pd.loc[i])):
                if self.data_loader.result_pd.at[i, j] == -1:
                    a = self.data_loader.encode_pd.at[i, j]
                    for k in a:
                        if isinstance(k, list):
                            if 100000 not in k:
                                total_cell += 1
                        else:
                            if k != 100000:
                                total_cell += 1
        return total_cell

    def layout(self, csv_path, plan_path, context=None):
        print(csv_path)
        self.data_loader.load_csv(csv_path, plan_path)

        self.layout_star()

        self.init()

        total_cell = self.calculate()

        # print("total_cell", total_cell, csv_path)

        display_clothing_dict, star = self.get_layout_clothing(self.shop_property_dict, total_cell)

        # self.allocation_cell_count(display_clothing_dict)

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

        self.data_loader.result_pd.to_csv(
            os.getcwd() + "/data/result/" + "%s.csv" % csv_path.split("/")[-1].split(".")[0])
        print("保存至[%s" % os.getcwd() + "/data/result/" + "%s.csv]" % csv_path.split("/")[-1].split(".")[0])

    def resort(self):
        fixed_settle = set()
        for name, value in self.data_loader.display_clothing_dict.items():
            if name.find("*") != -1:
                fixed_settle.add("男" if value[DataLoader.Sex_str] == 1 else "女" + value[DataLoader.Position_str][0])
            elif name.find("￥") != -1:
                fixed_settle.add(value[DataLoader.Position_str][1])
        print("fixed_settle", fixed_settle)

        # no wall
        for i in range(0, len(self.data_loader.result_pd.loc[0])):
            sorted_list = list()
            a = self.data_loader.result_pd[i]
            for j in range(len(a)):
                if isinstance(a[j], list) and "." not in self.data_loader.props_dict[j][i]:
                    sorted_list.append(a[j])
                    a[j] = [-1, a[j][0].split(":")[1]]
            sorted_list.sort(key=lambda a: a[1][0])
            for item in sorted_list:
                for j in range(len(a)):
                    if isinstance(a[j], list) and a[j][0] == -1:
                        a[j] = [item[0].split(":")[0]+":"+a[j][1], item[1], item[2]]
                        # a[j] = item[0].split(":")[0]+":"+a[j][1]
                        break

        ################## left wall ##################
        sorted_dict = dict()

        for i in range(0, int(len(self.data_loader.result_pd.loc[0])/2)):
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
                        a[j] = ["矩框", -1]
                    else:
                        a[j] = [-1]

        sorted_list = list(sorted_dict.items())

        sorted_list.sort(key=lambda a: a[1]["score"] + [10000 if "上衣" in a[1]["category"] else 0][0])
        print("sorted_list", sorted_list)

        for i in range(0, int(len(self.data_loader.result_pd.loc[0])/2)):
            a = self.data_loader.result_pd[i]
            for j in range(len(a)):
                if isinstance(a[j], list) and "." in self.data_loader.props_dict[j][i]:
                    # print(a[j], i, j)
                    item = sorted_list[0]

                    if a[j][0] == -1:
                        a[j] = item[0]
                    else:
                        a[j] = "/".join(["矩框", item[0]])
                        print(a[j])
                    item[1]["count"] -= 1
                    if item[1]["count"] < 1:
                        sorted_list.remove(item)
        print("sorted_list", sorted_list)

        # right wall
        sorted_dict = dict()
        for i in range(int(len(self.data_loader.result_pd.loc[0]) / 2), len(self.data_loader.result_pd.loc[0])):
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
                        a[j] = ["矩框", -1]
                    else:
                        a[j] = [-1]

        sorted_list = list(sorted_dict.items())
        sorted_list.sort(key=lambda a: a[1]["score"] + [10000 if "上衣" in a[1]["category"] else 0][0])
        print("sorted_list", sorted_list)

        for i in range(int(len(self.data_loader.result_pd.loc[0]) / 2), len(self.data_loader.result_pd.loc[0])):
            a = self.data_loader.result_pd[i]
            for j in range(len(a)):
                if isinstance(a[j], list) and "." in self.data_loader.props_dict[j][i]:
                    item = sorted_list[0]

                    if a[j][0] == -1:
                        a[j] = item[0]
                    else:
                        a[j] = "/".join(["矩框", item[0]])
                        print(a[j])

                    item[1]["count"] -= 1
                    if item[1]["count"] < 1:
                        sorted_list.remove(item)
        print("sorted_list", sorted_list)


if __name__ == "__main__":
    shop_property_csv_path = "/Users/quantum/code/StoreLayout/data/details/20181024/shop_property_utf8.csv"
    root_path = "/Users/quantum/code/StoreLayout/data/details/20181024/utf8_files/"
    plan_path = "/Users/quantum/code/StoreLayout/data/plan/2018_15_plan_utf8.csv"
    layout = LayoutManager(shop_property_csv_path)
    print(os.listdir(root_path))
    # exit()
    # ['西充.csv', '仁寿.csv', '高县.csv', '泸州.csv', '阆中.csv', '富顺.csv', '大英.csv', '雅安.csv', '绵阳.csv', '资阳.csv']
    for file_path in ['西充.csv', '仁寿.csv', '高县.csv', '泸州.csv', '阆中.csv', '富顺.csv', '大英.csv', '雅安.csv', '绵阳.csv', '资阳.csv']:
        layout.layout(os.path.join(root_path, file_path), plan_path, CONTEXT_DICT)
