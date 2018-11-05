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

    def alloc_pos(self, layout_clothing_dict):
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
        man_wall_cell_count = 0
        women_wall_cell_count = 0
        for i in self.data_loader.props_dict:
            middle = int(len(self.data_loader.encode_pd.loc[0])/2)
            for j in range(middle):
                if "." in self.data_loader.props_dict[i][j]:
                    count = 0
                    a = self.data_loader.encode_pd.at[i, j]
                    for k in a:
                        if 90 not in k:
                            count += 1
                    if sex_orientation == 1:
                        man_wall_cell_count += count
                    else:
                        women_wall_cell_count += count
            for j in range(middle, len(self.data_loader.encode_pd.loc[0])):
                if "." in self.data_loader.props_dict[i][j]:
                    count = 0
                    a = self.data_loader.encode_pd.at[i, j]
                    for k in a:
                        if 90 not in k:
                            count += 1
                    if sex_orientation == 1:
                        women_wall_cell_count += count
                    else:
                        man_wall_cell_count += count

        print("man_wall_cell_count", man_wall_cell_count)
        print("women_wall_cell_count", women_wall_cell_count)
        # print(self.data_loader.encode_pd)
        # exit()

        values = list(layout_clothing_dict.values())

        def wall_sort_value(a_value, sex):
            if "墙面alloc" not in a_value:
                a_value["墙面alloc"] = 0
            value = 0
            if sex == 1:
                value += a_value[DataLoader.Sex_str]*(10**10)
            else:
                value += (1 - a_value[DataLoader.Sex_str]) * (10 ** 10)

            value += 10 ** 11 if (a_value["current_cell_num"] - a_value["墙面alloc"]) > 0 else -10 ** 11

            if "墙面" == a_value[DataLoader.Position_str][0]:
                value += 10**9
            elif len(a_value[DataLoader.Position_str]) > 1 and "墙面" == a_value[DataLoader.Position_str][1]:
                value += 10**8

            # value += (10**4) * (a_value["current_cell_num"] - a_value["墙面alloc"])
            return value

        while man_wall_cell_count > 0:
            values.sort(key=lambda a: wall_sort_value(a, 1), reverse=True)
            values[0]["墙面alloc"] += 1
            man_wall_cell_count -= 1
        while women_wall_cell_count > 0:
            values.sort(key=lambda a: wall_sort_value(a, 0), reverse=True)
            values[0]["墙面alloc"] += 1
            women_wall_cell_count -= 1

    # MODEL NEED
    def internal_sort(self, clothing, context, index, column):

        depth = len(self.data_loader.encode_pd.index)

        def get_item_score(a):
            if "already_allocation_info" not in a:
                a["already_allocation_info"] = [0, []]
            if "wall_already_allocation_info" not in a:
                a["wall_already_allocation_info"] = 0
            score = sum([a["category_score_list"][i] * (10 ** 9 / (1000 ** i)) for i in range(
                len(a["category_score_list"]))])
            if a["current_cell_num"] - a["already_allocation_info"][0] > 0:
                score += 10**10
            score -= 20 ** math.log(max((datetime.now() - a[DataLoader.Execute_time_str]).days, 1))
            if "." in self.data_loader.props_dict[index][column] and "墙面alloc" in a and \
                    a["墙面alloc"] - a["wall_already_allocation_info"] > 0:
                score += 10**11
            elif "." not in self.data_loader.props_dict[index][column] and "墙面alloc" in a and \
                    a["墙面alloc"] - a["wall_already_allocation_info"] > 0:
                score -= 10**11

            if depth - index < depth/3 and a[DataLoader.Importance_str] in [4, 5]:
                score -= (10**10)*a[DataLoader.Importance_str]
            elif depth - index >= 2*depth/3 and a[DataLoader.Importance_str] in [4, 5]:
                score += (10**10)*a[DataLoader.Importance_str]
            if depth - index < depth/3 and a[DataLoader.Season_str] not in context["season"]:
                score -= 10**9

            return score

        clothing.sort(key=lambda a: get_item_score(a), reverse=True)

        return clothing

    def choose_best(self, section, boy_clothing, girl_clothing, context, index, column, rank=0):
        if section[1] < COLUMN_DIVIDE // 2 and sum([i["current_cell_num"]-i["already_allocation_info"][0] for i in boy_clothing]) > 0:
            clothing = boy_clothing
        elif section[1] >= COLUMN_DIVIDE // 2 and sum([i["current_cell_num"]-i["already_allocation_info"][0] for i in girl_clothing]) > 0:
            clothing = girl_clothing
        else:
            clothing = boy_clothing if sum([i["current_cell_num"]-i["already_allocation_info"][0] for i in boy_clothing]) > 0 else girl_clothing
        for i in clothing:
            i["category_score_list"] = [self.category_sort_info.get(j, {}).get(section[0], {}).get(section[1], 0) for j in
                                        i[DataLoader.Category_str]]

        clothing = self.internal_sort(clothing, context, index, column)
        if rank >= len(clothing):
            best_one = clothing[-1]
        else:
            best_one = clothing[rank]

        if "." in self.data_loader.props_dict[index][column] and "墙面alloc" in best_one and \
                best_one["墙面alloc"] - best_one["wall_already_allocation_info"] > 0:
            best_one["wall_already_allocation_info"] += 1

        if best_one["current_cell_num"] - best_one["already_allocation_info"][0] > 0:
            best_one["already_allocation_info"][0] += 1
        else:
            clothing.remove(best_one)

        clothing_name = best_one["name"]

        clothing_name = clothing_name.split("*")[0]
        clothing_name = clothing_name.split("￥")[0]

        return clothing_name

    def get_layout_clothing(self, plan, total_cell_count):
        print("before plan", plan)
        layout_clothing_dict = dict()
        star_clothing_dict = dict()
        display_clothing = copy.deepcopy(self.data_loader.display_clothing_dict)
        for i in display_clothing:
            if i.find("*星墙") != -1:
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
                    plan[self.Man_pants_str] -= value[DataLoader.Style_count_str][0]
                elif value[DataLoader.Category_str][0] in self.Pants_category_list and value[DataLoader.Sex_str] == 0:
                    # print("fu12", value[DataLoader.Style_str])
                    plan[self.Women_pants_str] -= value[DataLoader.Style_count_str][0]
                elif value[DataLoader.Category_str][0] in self.Upper_body_category_list and value[
                    DataLoader.Sex_str] == 0:
                    # print("fu13", value[DataLoader.Style_str])
                    plan[self.Woman_upper_body_str] -= value[DataLoader.Style_count_str][0]
                elif value[DataLoader.Category_str][0] in self.Upper_body_category_list and value[
                    DataLoader.Sex_str] == 1:
                    # print("fu14", value[DataLoader.Style_str])
                    plan[self.Man_upper_body_str] -= value[DataLoader.Style_count_str][0]

        print("after plan", plan)

        display_clothing_list = list()
        for i in display_clothing:
            if "count" not in display_clothing[i]:
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
            # print("fuck!!!!!fuck!!!!!fuck!!!!!", sum(plan.values()), plan[self.Man_upper_body_str],
            #       plan[self.Woman_upper_body_str], plan[self.Man_pants_str], plan[self.Women_pants_str])

            for i in display_clothing_list:
                # base_count = i[DataLoader.Style_str][0] if i[DataLoader.Style_str][0] > 0 else i[DataLoader.Style_str][1]
                base_count = int((i[DataLoader.Style_count_str][0] + i[DataLoader.Style_count_str][1]) / 2)
                i["threshold_value"] = min(math.ceil(self.Import_rate[i[DataLoader.Importance_str]] * base_count), 3)

                # if i["count"] + i["threshold_value"] > 8:
                #     i["threshold_value"] = 0
                if i["count"] + i["threshold_value"] > 6 and i["threshold_value"] > 0:
                    i["threshold_value"] = 1

                # print(i["threshold_value"], i["name"], i[DataLoader.Style_str], i["count"] + i["threshold_value"])
                if i[DataLoader.Importance_str] <= 2 and first_circle is True:
                    if i["name"].find("￥") != -1:
                        second_props_num = self.data_loader.second_props_num_dict.get(i["name"].split("￥")[-1], 0)
                        # print("second_props_num", second_props_num, i["name"])
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
                    if (display_clothing_list[i][DataLoader.Style_count_str][0] > 0 or
                        display_clothing_list[i][DataLoader.Style_count_str][1] > 0) and \
                            display_clothing_list[i][self.Man_upper_body_str] == 1 and \
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
                    if (display_clothing_list[i][DataLoader.Style_count_str][0] > 0 or
                        display_clothing_list[i][DataLoader.Style_count_str][1] > 0) and \
                            display_clothing_list[i][self.Man_pants_str] == 1 and \
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
                    if (display_clothing_list[i][DataLoader.Style_count_str][0] > 0 or
                        display_clothing_list[i][DataLoader.Style_count_str][1] > 0) and \
                            display_clothing_list[i][self.Woman_upper_body_str] == 1 and \
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
                    if (display_clothing_list[i][DataLoader.Style_count_str][0] > 0 or
                        display_clothing_list[i][DataLoader.Style_count_str][1] > 0) and \
                            display_clothing_list[i][self.Women_pants_str] == 1 and \
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
                        plan[self.Women_pants_str] -= 1
                        a["threshold_value"] -= 1
                        break
                if a is None:
                    break
        mean_cell_style_count = sum([value["count"] for value in display_clothing_list]) / total_cell_count

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
                            # print(name, "mean_cell_style_count", mean_cell_style_count, display["current_cell_num"], count)
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
                elif "*" in self.data_loader.props_dict[row][j] and "星中" in self.data_loader.props_dict[row][j]["*"]:
                    for display in display_clothing_list:
                        name = display["name"]
                        if name.find("星中") == -1:
                            continue
                        # print("*name", name)
                        count = display["count"]
                        sex = "男" if display[DataLoader.Sex_str] == 1 else "女"
                        props_full_name = name.split("*")[-1]
                        if sex not in props_full_name:
                            props_full_name = sex + props_full_name
                        if props_full_name == self.data_loader.props_dict[row][j]["*"]:
                            if "current_cell_num" not in display:
                                display["current_cell_num"] = 0
                            # print(name, "mean_cell_style_count", mean_cell_style_count,
                            #       display["current_cell_num"], count)
                            if display["current_cell_num"]*mean_cell_style_count < count:
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
                                # print("current_cell_num", display)
        # print(self.data_loader.result_pd)

        for i in display_clothing_list:
            layout_clothing_dict[i["name"]] = i
        return layout_clothing_dict, star_clothing_dict

    def allocation_cell_count(self, layout_clothing_dict):
        values = list(layout_clothing_dict.values())
        for value in values:
            if "current_cell_num" not in value:
                value["current_cell_num"] = 0
        total_cell = self.calculate() + sum([value["current_cell_num"] for value in values])
        print("total_cell", total_cell)
        mean_cell_style_count = sum([value["count"] for key, value in layout_clothing_dict.items()]) / total_cell
        remain_cell = total_cell - sum([value["current_cell_num"] for value in values])
        print("remain_cell", remain_cell, mean_cell_style_count)
        # remain_cell = 10  # test
        while remain_cell > 0:
            values.sort(key=lambda a: (a["count"] - a["current_cell_num"]*mean_cell_style_count)*100 + a[
                DataLoader.Importance_str], reverse=True)
            values[0]["current_cell_num"] += 1
            remain_cell -= 1
        print("all", sum([value["current_cell_num"] for value in values]))

        self.merge_residue(values, mean_cell_style_count)

        for value in values:
            # print("after", value["name"], value["count"], value["current_cell_num"])
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
                            if 90 not in k:
                                total_cell += 1
                        else:
                            if k != 90:
                                total_cell += 1
        return total_cell

    def layout(self, csv_path, plan_path, context=None):
        self.data_loader.load_csv(csv_path, plan_path)
        self.layout_star()

        total_cell = self.calculate()

        print("total_cell", total_cell, csv_path)

        display_clothing_dict, star = self.get_layout_clothing(self.shop_property_dict[csv_path.split("/")[-1].split(".")[0]], total_cell)

        print("sum", sum([display_clothing_dict[i]["count"] for i in display_clothing_dict]))

        self.allocation_cell_count(display_clothing_dict)

        self.alloc_pos(display_clothing_dict)

        boy_clothing = list()
        girl_clothing = list()
        for key, value in display_clothing_dict.items():
            # print("value", value)
            if "already_allocation_info" not in value:
                value["already_allocation_info"] = [0, []]
            if value["性别"] == 1:
                boy_clothing.append(value)
            else:
                girl_clothing.append(value)

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
                        encode_list = self.data_loader.encode_pd.at[index, column]
                        rank = 0
                        for k in encode_list:
                            if isinstance(k, list):
                                if 90 not in k:
                                    result_list.append(self.choose_best(
                                            section, boy_clothing, girl_clothing, context, index, column, rank))
                                    rank += 1
                                else:
                                    result_list.append("矩框")
                                    # rank += 1

                            else:
                                result_list.append(self.choose_best(section, boy_clothing, girl_clothing, context,
                                                                    index, column, rank))
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

                        return result_name

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
                        else:
                            break

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
        # exit()
