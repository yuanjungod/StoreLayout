import os
import pandas as pd
from lib.box_analyze_strategy import BoxAnalyseStragy
from lib.data_loader import DataLoader


class LayoutManager(object):

    def __init__(self):
        self.box_analyse = BoxAnalyseStragy()
        self.box_analyse_result = self.box_analyse.analyse(os.getcwd() + "/data/details/", 4)
        self.data_loader = DataLoader()

    def layout(self, csv_path):
        self.data_loader.load_csv(csv_path)
        total_cell = 0
        for i in self.data_loader.result_pd:
            for j in self.data_loader.result_pd[i]:
                if j == -1:
                    total_cell += 1
        if sum(layout.data_loader.display_clothing_dict.values()) != total_cell:
            print("可以摆放%s个单元格，提供了%s单元格服装" % (total_cell, sum(layout.data_loader.display_clothing_dict.values())))
            exit()


if __name__ == "__main__":
    layout = LayoutManager()
    layout.layout("/Users/happy/code/StoreLayout/data/details/雅安.csv")
    print(layout.data_loader.display_clothing_dict)








