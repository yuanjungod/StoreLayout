import os
from .data_loader import DataLoader


class BoxAnalyseStragy(object):

    def __init__(self):
        self.data_list = None
        self.box_gather = dict()

    def get_all_data(self, root_path):
        for i in os.listdir(root_path):
            data = DataLoader()
            data.load_csv(os.path.join(root_path, i))
            self.data_list.append(data)

    def get_all_box(self, root_path):
        if self.data_list is None:
            self.get_all_data(root_path)
        for encode_pd in self.data_list:
            print(encode_pd)