# -*- coding: utf-8 -*-
import os

root_path = "/Users/quantum/code/StoreLayout/data/result"
save_root_path = "/Users/quantum/code/StoreLayout/data/result_GB18030"

for origin_file in os.listdir(root_path):
    os.system("iconv -f UTF-8 -t GB18030 %s >%s" % (
        os.path.join(root_path, origin_file), os.path.join(save_root_path, origin_file)))

