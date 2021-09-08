import wget
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from pathlib import Path
import zipfile
import sys
import pdb
import ast



def download(url, savepath):
    wget.download(url, str(savepath))


def unzip(zippath, savepath):
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def get_count(tp, id):
    groups = tp[[id]].groupby(id, as_index=False)
    count = groups.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    for i, (_, group) in tqdm(enumerate(data_grouped_by_user), total=len(data_grouped_by_user)):
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def convert_to_strict_json(input_path, output_path):
    """
    input_path: .gz file
    output_path: json
    """
    import json
    import gzip

    def parse(path):
        g = gzip.open(path, 'r')
        for l in g:
            yield json.dumps(eval(l))

    f = open(output_path, 'w')
    for l in parse(input_path):
        f.write(l + '\n')
    return

# import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')




def Amazon(data_path, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # older Amazon
    # data_flie = '../../dataset/recommend/Amazon/reviews_' + dataset_name + '.json.gz'
    data_flie = data_path
    # latest Amazon
    # data_flie = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    item_set = set()
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        item_set.add(item) #构建item集合;
        datas.append((user, item, int(time)))
    return datas, item_set

def create_co_occurrence_matrix(data_path, occurrence_path, item_set):
    """
    extract the relationship from the amazon dataset, and treat the all relationships as the same one;

    extract "asin" and "related" columns;

    item_set: only extract items that are contained in the 'item_set', which is extracted from user behaviors.
    """
    # folder_path = self._get_rawdata_folder_path()
    # file_path = folder_path.joinpath('ratings_5_core.csv')
    file_path = data_path
    df = getDF(file_path)

    # pdb.set_trace()
    core_df = df[['asin','related']]
    max_len = -1
    min_len = 100000
    length_list = []
    with open(occurrence_path, 'w') as w_f:
        for index, row in core_df.iterrows():
            #判断是否在用户行为序列中, 不存在跳过;
            item_id = row['asin']
            if item_id not in item_set:
                continue
            data = [row['asin'], row['asin']] #自连边;
            # pdb.set_trace()
            # if not np.isnan(row['related']):
            if not (row['related'] is np.nan):
                dict_data = dict(row['related'])
                for key, value in dict_data.items():
                    value_del = [item for item in list(value) if item in item_set]
                    data.extend(list(value_del)) #暂时不考虑异构边;
            length_list.append(len(data))
            if max_len < len(data):
                max_len = len(data)
            
            if min_len > len(data):
                min_len = len(data)
            w_f.write(" ".join(data) + '\n')

    print("Max Len: {}, Min Len: {}, Mean Len: {}".format(max_len, min_len, np.mean(length_list)))
    return

def extract_item_category(data_path, item_category_path, item_set):
    df = getDF(data_path)

    core_df = df[['asin', 'categories']]
    # core_df = df

    with open(item_category_path, 'w') as w_f:
        for index, row in core_df.iterrows():
            item_id = row['asin']
            if item_id not in item_set:
                continue
            data = [row['asin']]
            if not (row['categories'] is np.nan):
                cate_list = list(row['categories'])[0]
                data.extend(cate_list) #第一个元素是item, 后续元素均是cates;
            w_f.write(" ".join(data) + '\n')
    return


def judge_pair_in_train_data(pair_path, train_path):
    pair_data = open(pair_path, 'r')
    train_data = open(train_path, 'r')

    #读取全部的pair
    all_pair_data = []
    df = getDF(pair_path)

    core_df = df[['asin','related']]

    occurrence_path = "all_pair_items.txt"
    with open(occurrence_path, 'w') as w_f:
        for index, row in core_df.iterrows():
            #判断是否在用户行为序列中, 不存在跳过;
            item_id = row['asin']
            # if item_id not in item_set:
            #     continue
            # data = [row['asin'], row['asin']] #自连边;
            # # pdb.set_trace()
            # # if not np.isnan(row['related']):
            if not (row['related'] is np.nan):
                dict_data = dict(row['related'])
                for key, value in dict_data.items():
                    value_del = [(item_id, item) for item in list(value)]
                    all_pair_data.extend(value_del)
                    # data.extend(list(value_del)) #暂时不考虑异构边;

    #构建全部的行为序列
    train_line = train_data.readline()
    train_all_data = []
    while train_line:
        train_one_data = train_line.strip().split()
        train_all_data.append(set(train_one_data))
        train_line = train_data.readline()

    #遍历全部序列
    item_pair_num = 0
    for pair_item in all_pair_data:
        for item_set in train_all_data:
            if set(pair_item).issubset(item_set):
                item_pair_num += 1
                break
    
    print("覆盖率: {}".format(item_pair_num*1.0 / len(all_pair_data)))
    return



if __name__ == "__main__":

    behavior_path = "/Users/a1234/Downloads/reviews_Beauty_5.json.gz"
    _, item_set = Amazon(behavior_path, rating_score=0.)
    
    # create_co_occurrence_matrix("/Users/a1234/Downloads/meta_Beauty.json.gz", "/Users/a1234/Downloads/beauty_graph.txt", item_set)
    # extract_item_category("/Users/a1234/Downloads/meta_Beauty.json.gz", "/Users/a1234/Downloads/item_category.txt", item_set)

    judge_pair_in_train_data()
    # data = getDF("/Users/a1234/Downloads/meta_Beauty.json.gz")
    # pdb.set_trace()
    # convert_to_strict_json("/Users/a1234/Downloads/meta_Beauty.json.gz", "/Users/a1234/Downloads/beauty.json")


