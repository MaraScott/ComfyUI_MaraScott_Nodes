#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###
    
import os 
import sys 
import shutil
import filecmp

from ... import __ROOT__file__
import __main__
import folder_paths as comfy_paths

def init():

    # sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__ROOT__file__)), "marasit_nodes_comfyui"))
    # sys.path.append(comfy_paths.base_path)

    _extensions_foler = "web" + os.sep + "extensions" + os.sep + "MarasIT"
    extentions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)), _extensions_foler)
    _javascript_folder = "web" + os.sep + "assets" + os.sep + "js"
    javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__ROOT__file__)), _javascript_folder )

    if not os.path.exists(extentions_folder):
        print('Making the "'+_extensions_foler+'" folder')
        os.mkdir(extentions_folder)

    result = filecmp.dircmp(javascript_folder, extentions_folder)

    if result.left_only or result.diff_files:
        print('Update to javascripts files detected')
        file_list = list(result.left_only)
        file_list.extend(x for x in result.diff_files if x not in file_list)

        for file in file_list:
            print(f'Copying {file} to extensions folder')
            src_file = os.path.join(javascript_folder, file)
            dst_file = os.path.join(extentions_folder, file)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_file)