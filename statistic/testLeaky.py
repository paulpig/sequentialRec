import numpy as np
import pdb

train_f = open('train_data.txt')
graph_f = open('new_cocurrence_correct.txt')

train_line = train_f.readline()
train_dict = {}
train_list_data = []
vocabulary_items = []
item2id = {}
while train_line:
    train_item = train_line.strip().split(" ")
    for item in train_item:
        if item not in item2id:
            item2id[item] = len(item2id)
        if item not in vocabulary_items:
            vocabulary_items.append(item)
    
    train_dict[item2id[train_item[0]]] = [item2id[item] for item in train_item[1:]] #第一个元素是test item
    train_list_data.append(set([item2id[item] for item in train_item[1:]])) #存放训练数据
    train_line = train_f.readline()

print("Finish Load train_data.")


graph_line = graph_f.readline()
graph_dict = {}
graph_array = np.zeros((len(vocabulary_items), len(vocabulary_items)))
while graph_line:
    graph_item = graph_line.strip().split(" ")
    # graph_dict[graph_item[0]] = graph_item[1:]
    output_token = graph_item[0]
    for token_s in graph_item[1:]:
        graph_array[item2id[output_token]][item2id[token_s]] = 1.0
        graph_array[item2id[token_s]][item2id[output_token]] = 1.0
    graph_line = graph_f.readline()

assert (graph_array == graph_array.transpose()).all()


print("Finish Load graph data.")
#train_dict: {test_item_id: train_items_list}
for test_item, train_ids in train_dict.items():
    graph_co_items = np.where(graph_array[test_item] > 0.0)[0]
    graph_co_items_column = np.where(graph_array[:][test_item] > 0.0)[0]
    assert set(graph_co_items) == set(graph_co_items_column)

    item_valid  = list(set(train_ids) & set(graph_co_items))
    #是否出现在训练集中
    valid_number = 0
    for item_s in item_valid:
        for train_list_items in train_list_data:
            if set([test_item, item_s]).issubset(train_list_items):
                valid_number += 1
                break
    
    if len(item_valid) != valid_number:
        print("信息泄露了!!!!!")
        pdb.set_trace()

print("success!!!!")