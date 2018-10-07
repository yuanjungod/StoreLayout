import os
from data_loader import DataLoader


class BoxAnalyseStragy(object):

    def __init__(self):
        self.data_list = None
        self.box_gather = dict()

    @classmethod
    def get_value_index_column(cls, encode_pd, value):
        result_list = list()
        for i in encode_pd.index:
            for j in range(len(encode_pd.loc[i])):
                if value == encode_pd.at[i, j]:
                    result_list.append((i, j))
        return result_list

    @classmethod
    def get_divide_partition(cls, total_length, divide_count=4):
        result_list = list()
        unit_length = total_length // (divide_count + 1)
        for i in range(divide_count + 1):
            result_list.append(i * unit_length)
        result_list.append(total_length)
        return result_list

    def get_all_data(self, root_path):
        self.data_list = list()
        for i in os.listdir(root_path):
            print("###################################", i)
            data = DataLoader()
            data.load_csv(os.path.join(root_path, i))
            self.data_list.append(data)

    def get_all_box(self, root_path, divide_count):
        result_box_dict = {}
        if self.data_list is None:
            self.get_all_data(root_path)
        for encode_pd in self.data_list:
            index_divide = self.get_divide_partition(encode_pd.encode_pd.shape[0], divide_count=4)
            column_divide = self.get_divide_partition(encode_pd.encode_pd.shape[1], divide_count=4)
            cash_orientation = self.get_value_index_column(encode_pd.encode_pd, 10009)
            if sum([i[1] for i in cash_orientation]) / len(cash_orientation) < len(encode_pd.encode_pd.loc[0]) / 2:
                for index in range(divide_count):
                    if index not in result_box_dict:
                        result_box_dict[index] = dict()
                    for column in range(divide_count):
                        if column not in result_box_dict[index]:
                            result_box_dict[index][column] = list()
                        result_box_dict[index][column].append(
                            encode_pd.encode_pd.loc[index_divide[divide_count - 1 - index]:
                                                    index_divide[divide_count - index]][column_divide[column]:
                                                                                        column_divide[column + 1]])
        return result_box_dict

    def analyse(self, root_path, divide_count):
        box_analyse_result = dict()
        result_box_dict = self.get_all_box(root_path, divide_count)
        category_dict = self.data_list[0].category_dict
        for i in range(divide_count):
            if i not in box_analyse_result:
                box_analyse_result[i] = dict()
            for j in range(divide_count):
                if j not in box_analyse_result[i]:
                    box_analyse_result[i][j] = dict()
                    tmp_pd = result_box_dict[i][j]
                    for k in tmp_pd.index:
                        for h in range(len(tmp_pd[k])):
                            category_info = category_dict[tmp_pd.loc[k][h]]
                            for item in ["sex", "season", "date", "category"]:
                                if item == "category":
                                    category_list = category_info[item].split("/")
                                    for category in category_list:
                                        if category not in box_analyse_result[i][j]:
                                            box_analyse_result[i][j][category] = 0
                                        box_analyse_result[i][j][category] += 1
                                else:
                                    if category_info[item] not in box_analyse_result[i][j]:
                                        box_analyse_result[i][j][category_info[item]] = 0
                                    box_analyse_result[i][j][category_info[item]] += 1


if __name__ == "__main__":
    box_analyse = BoxAnalyseStragy()
    print(box_analyse.get_divide_partition(12, 3))
    result_box_dict = box_analyse.get_all_box(os.getcwd() + "/../data/details/", 4)
    for i in result_box_dict:
        for j in result_box_dict[i]:
            print("#####", result_box_dict[i][j])
