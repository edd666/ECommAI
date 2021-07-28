# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021/7/27
# @Contact : liaozhi_edo@163.com


def _itemcf_sim_in_multiprocess(args):
    """
    计算物品相似度-多进程版本的内置函数

    :param args: tuple 参数
    :return:
    """
    user_item_time_dict, location_weight, time_weight, normalization, item_cnt_queue, i2i_sim_queue = args

    item_cnt, i2i_sim = itemcf_sim(user_item_time_dict, location_weight, time_weight, normalization)

    item_cnt_queue.put(item_cnt)
    i2i_sim_queue.put(i2i_sim)

    return


def itemcf_sim_in_multiprocess(user_item_time_dict, location_weight=False, time_weight=False):
    """
    计算物品相似度-多进程版本

    注意: user_item_time_dict中行为已按时间升序排列.

    :param user_item_time_dict: dict 用户行为序列,形如{u:[(i,t)]},其中t为时间戳,单位为s
    :param location_weight: bool 是否采用位置加权
    :param time_weight: bool 是否采用时间加权
    :return:
        i2i_sim: dict 物品相似度
    """
    # 1,初始化
    users = list(user_item_time_dict.keys())
    num_cpus = mp.cpu_count() // 2 if len(users) > 30000 else 1
    batch_size = len(users) // num_cpus

    # 2,计算物品相似度和频次(不做热度降权)
    pool = mp.Pool(num_cpus)
    item_cnt_queue = mp.Manager().Queue()
    i2i_sim_queue = mp.Manager().Queue()
    for idx in range(0, len(users), batch_size):
        b_dict = dict()
        for user in users[idx:idx + batch_size]:
            b_dict[user] = user_item_time_dict[user]
        args = (b_dict, location_weight, time_weight, False, item_cnt_queue, i2i_sim_queue)  # normalization=False
        pool.apply_async(_itemcf_sim_in_multiprocess, (args,))

    pool.close()
    pool.join()

    # 合并
    item_cnt = defaultdict(int)  # item count
    while not item_cnt_queue.empty():
        item_cnt_ = item_cnt_queue.get()
        if not isinstance(item_cnt_, dict):
            raise ValueError('item_cnt must be a dict')
        for item in item_cnt_:
            item_cnt[item] += item_cnt_[item]

        del item_cnt_

    i2i_sim = defaultdict(dict)  # item sim
    while not i2i_sim_queue.empty():
        i2i_sim_ = i2i_sim_queue.get()
        if not isinstance(i2i_sim_, dict):
            raise ValueError('i2i_sim must be a dict')
        for item, related_items in i2i_sim_.items():
            for related_item, score in related_items.items():
                i2i_sim[item].setdefault(related_item, 0.0)
                i2i_sim[item][related_item] += score

        del i2i_sim_

    # 3,热度降权
    for item, related_items in i2i_sim.items():
        for related_item, score in related_items.items():
            i2i_sim[item][related_item] = score / np.sqrt(item_cnt[item] * item_cnt[related_item])

    return i2i_sim

