import os
from lib.box_analyze_strategy import BoxAnalyseStragy


class LayoutManager(object):

    def __init__(self):
        self.box_analyse = BoxAnalyseStragy()
        self.box_analyse_result = self.box_analyse.analyse(os.getcwd() + "/data/details/", 4)

    def layout(self, store):
        pass








