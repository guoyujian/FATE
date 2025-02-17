#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

from fate_arch.common import file_utils


# def get_data_table_count(path):
#     config_path = os.path.join(path, "config.yaml")
#     config = file_utils.load_yaml_conf(conf_path=config_path)
#     count = 0
#     if config:
#         if config.get("type") != "vision":
#             raise Exception(f"can not support this type {config.get('type')}")
#         ext = config.get("inputs").get("ext")
#         base_dir = os.path.join(path, "images")
#         for file_name in os.listdir(base_dir):
#             if file_name.endswith(ext):
#                 count += 1
#     return count
def get_data_table_count(path):
    config_path = os.path.join(path, "config.yaml")
    config = file_utils.load_yaml_conf(conf_path=config_path)
    count = 0
    if config:
        if config.get("type") != "vision" and config.get("type") != "nlp" and config.get("type") != "mm" :
            raise Exception(f"can not support this type {config.get('type')}")
        if config.get("type") == "vision":
            ext = config.get("inputs").get("ext")
            base_dir = os.path.join(path, "images")
            for file_name in os.listdir(base_dir):
                if file_name.endswith(ext):
                    count += 1
        if config.get("type") == "nlp" or config.get("type") == 'mm':
            count = config.get("inputs").get("count")

    return count