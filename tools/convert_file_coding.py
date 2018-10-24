# -*- coding: utf-8 -*-
import os

root_path = "/Users/quantum/code/StoreLayout/data/details/20181024/origin_files"
save_root_path = "/Users/quantum/code/StoreLayout/data/details/20181024/utf8_files"

for origin_file in os.listdir(root_path):
    os.system("iconv -f GB18030 -t UTF-8 %s >%s" % (
        os.path.join(root_path, origin_file), os.path.join(save_root_path, origin_file)))

