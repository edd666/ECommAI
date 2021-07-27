# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021/7/26
# @Contact : liaozhi_edo@163.com


"""
    Utils
"""

# packages
import torch
import random
import numpy as np


def reduce_memory_usage(df):
    """
    iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.

    :param df: DataFrame
    :return:
        df: DataFrame
    """
    # `deep=True`返回实际内存占用
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def setup_seed(seed):
    """
    设置随机种子

    :param seed: int 随机种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    return


def build_user_behavior_sequence(df):
    """
    构建用户行为序列

    :param df: DataFrame 用户行为
    :return:
        user_item_time_dict: dict 用户行为序列,形如{u:[(i, t)]}
    """
    # 1,排序
    df.sort_values(by=['timestamp'], ascending=True, inplace=True)

    def make_item_time_pair(group):
        return list(zip(group['item_id'], group['timestamp']))

    # 2,构建序列
    user_item_time_df = df.groupby(by=['user_id'])[['item_id', 'timestamp']] \
        .apply(lambda x: make_item_time_pair(x)) \
        .reset_index(drop=False) \
        .rename(columns={0: 'item_time_list'})

    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict
