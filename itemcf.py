# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021/7/27
# @Contact : liaozhi_edo@163.com


"""
    ItemCF
"""

# packages
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def itemcf_sim(user_item_time_dict, location_weight=False, time_weight=False, normalization=True):
    """
    计算物品相似度

    注意: user_item_time_dict中行为已按时间升序排列.

    :param user_item_time_dict: dict 用户行为序列,形如{u:[(i,t)]},其中t为时间戳,单位为s
    :param location_weight: bool 是否采用位置加权
    :param time_weight: bool 是否采用时间加权
    :param normalization: bool 是否采用热度降权
    :return:
        item_cnt: dict 物品频次
        i2i_sim: dict 物品相似度
    """
    # 1,物品相似度
    item_cnt = defaultdict(int)  # item count
    i2i_sim_ = defaultdict(dict)  # item smi 临时变量
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for loc1, (item1, time1) in enumerate(item_time_list):
            item_cnt[item1] += 1
            for loc2, (item2, time2) in enumerate(item_time_list):
                # 同一个物品
                if item1 == item2:
                    continue

                # 位置权重
                if location_weight:
                    loc_alpha = 1 if loc1 > loc2 else 0.7
                    loc_weight = loc_alpha * (0.9 ** (np.abs(loc1 - loc2) - 1))
                else:
                    loc_weight = 1

                # 时间权重
                if time_weight:
                    if isinstance(time1, int):
                        t_weight = np.exp(0.7 ** (np.abs(time1 - time2)))
                    else:
                        raise ValueError('time in user_item_time_dict must be int')
                else:
                    t_weight = 1

                # 相似度
                i2i_sim_[item1].setdefault(item2, 0.0)
                i2i_sim_[item1][item2] += loc_weight * t_weight / np.log(len(item_time_list) + 1)

    # 2,热度降权
    i2i_sim = i2i_sim_.copy()
    if normalization:
        for item, related_items in i2i_sim_.items():
            for related_item, score in related_items.items():
                i2i_sim[item][related_item] = score / np.sqrt(item_cnt[item] * item_cnt[related_item])
        return i2i_sim
    else:
        return item_cnt, i2i_sim


def evaluation(user_item_time_dict, i2i_sim, valid_user_item_time_dict):
    # 1,初始化
    users = list(valid_user_item_time_dict.keys())
    num_cpus = 10
    batch_size = len(users) // num_cpus

    pass