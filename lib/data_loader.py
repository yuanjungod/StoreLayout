# _*_ coding: utf-8 _*-
import pandas as pd
import random
import datetime
import os
from lib.config import *

random.seed(1)


class DataLoader(object):

    Sex_dict = {"男": 1, "女": 0}
    Season_dict = {"夏": 35, "冬": 2, "深秋冬": 10, "春秋": 20, "深秋": 15}

    Sex_str = "性别"
    Season_str = "季节"
    Execute_time_str = "执行时间"
    Category_str = "类型"
    Style_count_str = "款数"
    Importance_str = "重要程度（1-5级别)"
    Position_str = "位置"

    Date_format_str = "%Y年%m月%d日"

    def __init__(self):
        self.origin_pd = None
        self.etl_pd = None
        self.result_pd = None
        self.encode_pd = None
        self.clothing_property = None
        self.display_clothing = None
        self.category_dict = dict()
        self.name_dict = dict()
        self.display_clothing_dict = dict()
        self.props_dict = dict()
        self.second_props_num_dict = dict()
        self.props_item_dict = dict()
        self.file_name = None

    def load_csv(self, origin_csv_path, plan_csv_path=None):

        self.origin_pd = None
        self.etl_pd = None
        self.result_pd = None
        self.encode_pd = None
        self.clothing_property = None
        self.display_clothing = None
        self.category_dict = dict()
        self.name_dict = dict()
        self.props_dict = dict()
        self.display_clothing_dict = dict()
        self.second_props_num_dict = dict()
        self.file_name = origin_csv_path.split("/")[-1].split(".")[0]

        self.clothing_property = pd.read_csv(os.getcwd() + "/data/details/20181024/clothing_property_utf8.csv")
        # print(self.clothing_property.loc[0])
        for i in self.clothing_property.index:
            # print(i, self.clothing_property.loc[i][0])
            self.category_dict[i] = {
                "name": self.clothing_property.loc[i][1],
                "sex": self.Sex_dict[self.clothing_property.loc[i][2]],
                "season": self.Season_dict[self.clothing_property.loc[i][3]],
                "date": datetime.datetime.strptime(self.clothing_property.loc[i][4], self.Date_format_str),
                "category": self.clothing_property.loc[i][5].split("&")
            }
            self.name_dict[self.clothing_property.loc[i][1]] = {
                "index": i,
                "sex": self.Sex_dict[self.clothing_property.loc[i][2]],
                "season": self.Season_dict[self.clothing_property.loc[i][3]],
                "date": datetime.datetime.strptime(self.clothing_property.loc[i][4], self.Date_format_str),
                "category": self.clothing_property.loc[i][5].split("&")
            }

        if plan_csv_path is not None:
            self.display_clothing = pd.read_csv(plan_csv_path)
            # print(self.display_clothing.loc[0]["品名"])

            for i in self.display_clothing.index:
                name = "%s%s" % (self.display_clothing.loc[i]["性别"], self.display_clothing.loc[i]["品名"])
                self.display_clothing_dict[name] = dict()
                if name.find("*") != -1:
                    # self.display_clothing_dict[name.split("*")[0]] = dict()
                    self.display_clothing_dict[name]["*"] = 1
                    # name = name.split("*")[0]
                elif name.find("￥") != -1:
                    # self.display_clothing_dict[name.split("￥")[0]] = dict()
                    self.display_clothing_dict[name]["￥"] = 1
                    # name = name.split("￥")[0]

                for plan_property in [self.Sex_str, self.Season_str, self.Execute_time_str, self.Category_str,
                                      self.Position_str, self.Style_count_str, self.Importance_str]:
                    if plan_property == self.Sex_str:
                        self.display_clothing_dict[name][plan_property] = self.Sex_dict[
                            self.display_clothing.loc[i][plan_property]]
                    elif plan_property == self.Execute_time_str:
                        self.display_clothing_dict[name][plan_property] = datetime.datetime.strptime(
                            self.display_clothing.loc[i][plan_property], self.Date_format_str)
                    elif plan_property == self.Category_str:
                        self.display_clothing_dict[name][plan_property] = self.display_clothing.loc[i][
                            plan_property].split("&")
                    elif plan_property == self.Position_str:
                        position = self.display_clothing.loc[i][plan_property]
                        if position.find("@") != -1:
                            position = position.split("@")
                        elif position.find("￥") != -1:
                            position = position.split("￥")
                        else:
                            position = [position]
                        self.display_clothing_dict[name][plan_property] = position
                    elif plan_property == self.Style_count_str:
                        if len(self.display_clothing.loc[i][plan_property].split("-")) == 2:
                            self.display_clothing_dict[name][plan_property] = [int(i) for i in self.display_clothing.loc[i][plan_property].split("-")]
                        else:
                            self.display_clothing_dict[name][plan_property] = [
                                int(self.display_clothing.loc[i][plan_property]), int(self.display_clothing.loc[i][plan_property])]
                    elif plan_property == self.Importance_str:
                        self.display_clothing_dict[name][plan_property] = self.display_clothing.loc[i][plan_property]
                    elif plan_property == self.Season_str:
                        self.display_clothing_dict[name][plan_property] = self.display_clothing.loc[i][plan_property]
            # print(self.display_clothing_dict)
            # exit()

        self.origin_pd = pd.read_csv(origin_csv_path)
        result_list = list()
        for i in self.origin_pd.index:
            # if not (pd.isna(self.origin_pd.loc[i][0]) or pd.isnull(pd.isna(self.origin_pd.loc[i][0]))):
            item_list = self.origin_pd.loc[i].tolist()
            result_list.append(item_list)
            # print(i, item_list)

        self.etl_pd = pd.DataFrame(result_list, columns=[i for i in range(len(self.origin_pd.loc[0]))])
        self.etl_pd = self.etl_pd.fillna(0)

        self.result_pd = self.etl_pd.copy()
        self.encode_pd = self.etl_pd.copy()
        for i in self.etl_pd.index:
            self.props_dict[i] = dict()
            for j in range(len(self.etl_pd.loc[i])):
                self.props_dict[i][j] = dict()
                # print("before", self.etl_pd.loc[i][j], self.encode_pd.loc[i][j])
                if str(self.etl_pd.loc[i][j]).find("￥") != -1:
                    self.props_dict[i][j]["￥"] = str(self.etl_pd.loc[i][j]).split("￥")[-1]

                    if self.props_dict[i][j]["￥"] not in self.second_props_num_dict:
                        self.second_props_num_dict[self.props_dict[i][j]["￥"]] = 0
                    self.second_props_num_dict[self.props_dict[i][j]["￥"]] += len(str(self.etl_pd.loc[i][j]).split("/"))

                    self.etl_pd.at[i, j] = str(self.etl_pd.loc[i][j]).split("￥")[0]
                    self.encode_pd.at[i, j] = str(self.etl_pd.loc[i][j]).split("￥")[0]

                if str(self.etl_pd.loc[i][j]).find("*") != -1:
                    self.props_dict[i][j]["*"] = str(self.etl_pd.loc[i][j]).split("*")[-1]

                    if self.props_dict[i][j]["*"] not in self.second_props_num_dict:
                        self.second_props_num_dict[self.props_dict[i][j]["*"]] = 0
                    self.second_props_num_dict[self.props_dict[i][j]["*"]] += len(str(self.etl_pd.loc[i][j]).split("/"))

                    self.etl_pd.at[i, j] = str(self.etl_pd.loc[i][j]).split("*")[0]
                    self.encode_pd.at[i, j] = str(self.etl_pd.loc[i][j]).split("*")[0]
                if str(self.etl_pd.at[i, j]).find(":") != -1:
                    self.props_dict[i][j][":"] = str(self.etl_pd.loc[i][j]).split(":")[-1]

                    if self.props_dict[i][j][":"] not in self.second_props_num_dict:
                        self.second_props_num_dict[self.props_dict[i][j][":"]] = 0
                    self.second_props_num_dict[self.props_dict[i][j][":"]] += len(str(self.etl_pd.loc[i][j]).split("/"))

                    self.etl_pd.at[i, j] = str(self.etl_pd.loc[i][j]).split(":")[0]
                    self.encode_pd.at[i, j] = str(self.etl_pd.loc[i][j]).split(":")[0]

                # print("after", str(self.etl_pd.loc[i][j]).find(":") != -1, self.encode_pd.loc[i][j])

                category_info_list = list()
                for k in str(self.etl_pd.loc[i][j]).split("/"):
                    category_info_list.extend(k.split("&"))
                common_set = set(category_info_list) & set(self.name_dict.keys())
                if self.etl_pd.loc[(i, j)] in self.name_dict or len(list(common_set)) > 0:
                    self.result_pd.loc[(i, j)] = -1

                    self.encode_pd.at[i, j] = []
                    for m in str(self.etl_pd.loc[(i, j)]).split("/"):
                        self.encode_pd.loc[(i, j)].append([self.name_dict[n]["index"] for n in m.split("&")])
                    if len(self.props_dict[i][j]) == 0:
                        self.props_dict[i][j]["."] = ["墙面"]
                elif self.etl_pd.loc[(i, j)] in ENTITY_DICT:
                    # self.result_pd.at[i, j] = ENTITY_DICT[self.etl_pd.loc[i][j]]
                    self.encode_pd.at[i, j] = ENTITY_DICT[self.etl_pd.loc[i][j]]
                else:
                    if isinstance(self.etl_pd.loc[(i, j)], str) and self.etl_pd.loc[i][j] != "0":
                        print(self.etl_pd.loc[i][j], type(self.etl_pd.loc[i][j]), self.etl_pd.loc[i][j].find(":") != -1)
                    self.result_pd.at[(i, j)] = 0
                    self.encode_pd.at[(i, j)] = 0



