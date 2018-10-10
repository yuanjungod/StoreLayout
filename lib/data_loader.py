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

    def load_csv(self, origin_csv_path, save_path=None):

        self.clothing_property = pd.read_csv(os.getcwd() + "/data/clothing_property.csv")
        for i in self.clothing_property.index:
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

        self.display_clothing = pd.read_csv(os.getcwd() + "/data/display_clothering.csv", )
        for i in self.display_clothing.index:
            self.display_clothing_dict[self.display_clothing.loc[i][0]] = self.display_clothing.loc[i][1]

        self.origin_pd = pd.read_csv(origin_csv_path)
        result_list = list()
        for i in self.origin_pd.index:
            if not (pd.isna(self.origin_pd.loc[i][0]) or pd.isnull(pd.isna(self.origin_pd.loc[i][0]))):
                item_list = self.origin_pd.loc[i].tolist()
                result_list.append(item_list)

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

    def generate_train_data(self, origin_csv_path):
        category_set = set()
        all = list()
        for i in self.etl_pd.index:
            for j in range(len(self.etl_pd.loc[i])):
                print(self.etl_pd.loc[i][j], type(self.etl_pd.loc[i][j]))
                if str(self.etl_pd.loc[i][j]) not in ["橱窗", "入口", "推广", "试衣间", "0.0", "0"]:
                    # print("%s: %s" % (self.etl_pd.loc[i][j], count))
                    category_set.add(self.etl_pd.loc[i][j])
                    all.append(self.etl_pd.loc[i][j])
        count = 0
        for category in category_set:
            print("'%s': %s," % (category, count))
            count += 1
        print(len(all))

        # train_x = list()
        # train_y = list()
        #
        # if self.etl_pd is not None:
        #     self.load_csv(origin_csv_path)
        #
        # for i in self.etl_pd.index:
        #     for j in range(len(self.etl_pd.loc[i])):
        #         if self.etl_pd.loc[i][j] in category_dict:
        #             pass

        return self.etl_pd, self.result_pd

    @classmethod
    def data_generate1(cls, batch_size):
        simulated_data = list()
        # simulated_data = [0 for i in range(10)]
        # simulated_data.append(1)
        # simulated_data.append(1)
        # simulated_data.append(1)
        for i in range(1000):
            tmp = random.choice(range(OUT_DIM))
            if random.random() < 0.6:
                simulated_data.append(tmp)
                simulated_data.append(tmp)
                simulated_data.append(tmp)
                simulated_data.append(tmp)
                simulated_data.append(tmp)
            else:
                simulated_data.append(tmp)
                simulated_data.append(tmp)
                simulated_data.append(tmp)
                simulated_data.append(tmp)
                simulated_data.append(tmp)
                simulated_data.append(tmp)
            if len(simulated_data) >= 1000:
                simulated_data = simulated_data[:1000]
                break

        train_x = list()
        train_y = list()

        while True:
            shuffle_index_list = [i for i in range(890)]
            random.shuffle(shuffle_index_list)
            for i in shuffle_index_list:
                train_x.append(simulated_data[i: i+IN_DIME])
                train_y.append(simulated_data[i+IN_DIME+1])
                if len(train_x) == batch_size:
                    train_x = torch.unsqueeze(torch.FloatTensor(train_x), 0)
                    train_x = train_x.view(-1, 1, IN_DIME)

                    train_y = torch.LongTensor(train_y)
                    yield train_x, train_y

                    train_x = list()
                    train_y = list()


if __name__ == "__main__":
    # data_iter = DataLoader.data_generate1(10)
    # print(next(data_iter))
    print(os.getcwd())
    dl = DataLoader()
    dl.load_csv(os.getcwd() + "/../data/details/仁寿.csv", "../data/云货架门店摆放etl.csv")
    print(dl.result_pd)
    print(dl.encode_pd)

    # print(pd1.loc[0][0])
    # print(dl.generate_train_data("/Users/happy/code/StoreLayout/data/云货架门店摆放.csv"))
    # import os
    # category_set = set()
    # for i in os.listdir("/Users/happy/code/StoreLayout/data/details"):
    #     result_list = list()
    #     pd_data = pd.read_csv("/Users/happy/code/StoreLayout/data/details/" + i)
    #     for j in pd_data.index:
    #         if not (pd.isna(pd_data.loc[j][0]) or pd.isnull(pd.isna(pd_data.loc[j][0]))):
    #             item_list = pd_data.loc[j].tolist()
    #             # current_item = item_list[0]
    #             # for j in range(len(item_list)):
    #             #     if pd.isna(item_list[j]):
    #             #         item_list[j] = current_item
    #             #     else:
    #             #         current_item = item_list[j]
    #             result_list.append(item_list)
    #             etl_pd = pd.DataFrame(result_list, columns=[n for n in range(len(pd_data.loc[0]))])
    #             for k in etl_pd.index:
    #                 for y in etl_pd.loc[k]:
    #                     if not (pd.isna(y) or pd.isnull(y) or isinstance(y, np.int64) or isinstance(y, np.float64)):
    #                         # print(y, )
    #                         list1 = y.split("/")
    #                         # print(list1)
    #                         for m in list1:
    #                             for j1 in m.split("&"):
    #                                 if j1 == "裤":
    #                                     print(y)
    #                                 category_set.add(j1)
    # count = 0
    # category_set.remove("0")
    # for item in category_set:
    #     print(item)
    #     count += 1


