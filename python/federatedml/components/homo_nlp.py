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

from .components import ComponentMeta

homo_nlp_cpn_meta = ComponentMeta("HomoNLP")


@homo_nlp_cpn_meta.bind_param
def homo_nlp_param():
    from federatedml.param.homo_nlp_param import HomoNLPParam

    return HomoNLPParam


@homo_nlp_cpn_meta.bind_runner.on_guest.on_host
def homo_nlp_runner_client():
    from federatedml.nn.homo_nlp.enter_point import HomoNLPDefaultClient

    return HomoNLPDefaultClient


@homo_nlp_cpn_meta.bind_runner.on_arbiter
def homo_nlp_runner_arbiter():
    from federatedml.nn.homo_nlp.enter_point import HomoNLPDefaultServer

    return HomoNLPDefaultServer
