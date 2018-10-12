import pandas as pd
import random
import torch
import datetime
import os
from lib.config import *

random.seed(1)


class DataLoader(object):

    Sex_dict = {"男": 1, "女": 0}
    Season_dict = {"春秋": 0, "夏": 1, "冬": 2}

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
        self.file_name = None

    def load_csv(self, origin_csv_path):

        self.origin_pd = None
        self.etl_pd = None
        self.result_pd = None
        self.encode_pd = None
        self.clothing_property = None
        self.display_clothing = None
        self.category_dict = dict()
        self.name_dict = dict()
        self.display_clothing_dict = dict()
        self.file_name = origin_csv_path.split("/")[-1].split(".")[0]

        self.clothing_property = pd.read_csv(os.getcwd() + "/data/clothing_property.csv")
        for i in self.clothing_property.index:
            # print(i, self.clothing_property.loc[i][0])
            self.category_dict[i] = {
                "name": self.clothing_property.loc[i][0],
                "sex": self.Sex_dict[self.clothing_property.loc[i][1]],
                "season": self.Season_dict[self.clothing_property.loc[i][2]],
                "date": datetime.datetime.strptime(self.clothing_property.loc[i][3], "%Y/%m/%d"),
                "category": self.clothing_property.loc[i][4].split("/")
            }
            self.name_dict[self.clothing_property.loc[i][0]] = {
                "index": i,
                "sex": self.Sex_dict[self.clothing_property.loc[i][1]],
                "season": self.Season_dict[self.clothing_property.loc[i][2]],
                "date": datetime.datetime.strptime(self.clothing_property.loc[i][3], "%Y/%m/%d"),
                "category": self.clothing_property.loc[i][4].split("/")
            }

        self.display_clothing = pd.read_csv(os.getcwd() + "/data/display_clotheringUTF-8.csv", )
        for i in self.display_clothing.index:
            # self.display_clothing_dict[self.display_clothing.loc[i][0]] = self.display_clothing.loc[i][1]
            # print(i, self.display_clothing["Unnamed: 0"][i], self.display_clothing[self.file_name][i])
            self.display_clothing_dict[self.display_clothing["Unnamed: 0"][i]] = self.display_clothing[self.file_name][i]

        self.origin_pd = pd.read_csv(origin_csv_path)
        result_list = list()
        for i in self.origin_pd.index:
            if not (pd.isna(self.origin_pd.loc[i][0]) or pd.isnull(pd.isna(self.origin_pd.loc[i][0]))):
                item_list = self.origin_pd.loc[i].tolist()
                result_list.append(item_list)
            # print(i, item_list)

        self.etl_pd = pd.DataFrame(result_list, columns=[i for i in range(len(self.origin_pd.loc[0]))])
        self.etl_pd = self.etl_pd.fillna(0)

        self.result_pd = self.etl_pd.copy()
        self.encode_pd = self.etl_pd.copy()
        for i in self.etl_pd.index:
            for j in range(len(self.etl_pd.loc[i])):
                category_info_list = list()
                for k in str(self.etl_pd.loc[i][j]).split("/"):
                    category_info_list.extend(k.split("&"))
                common_set = set(category_info_list) & set(self.name_dict.keys())
                if self.etl_pd.loc[(i, j)] in self.name_dict or len(list(common_set)) > 0:
                    self.result_pd.loc[(i, j)] = -1

                    self.encode_pd.at[i, j] = []
                    for m in str(self.etl_pd.loc[(i, j)]).split("/"):
                        self.encode_pd.loc[(i, j)].append([self.name_dict[n]["index"] for n in m.split("&")])
                elif self.etl_pd.loc[(i, j)] in ENTITY_DICT:
                    self.result_pd.at[i, j] = ENTITY_DICT[self.etl_pd.loc[i][j]]
                    self.encode_pd.at[i, j] = ENTITY_DICT[self.etl_pd.loc[i][j]]
                else:
                    if isinstance(self.etl_pd.loc[(i, j)], str) and self.etl_pd.loc[i][j] != "0":
                        print(self.etl_pd.loc[i][j], type(self.etl_pd.loc[i][j]))
                    self.result_pd.at[(i, j)] = 0
                    self.encode_pd.at[(i, j)] = 0



