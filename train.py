from lambdamart import LambdaMART
import numpy as np
import pandas as pd
import json
import  os



def get_data(file_loc):

    return np.array(genSamples(file_loc))

def group_queries(data):
    query_indexes = {}
    index = 0
    for record in data:
        query_indexes.setdefault(record[1], [])
        query_indexes[record[1]].append(index)
        index += 1
        return query_indexes




# train_path=["part-00000","part-00001","part-00002","part-00003","part-00004","part-00005","part-00006","part-00007"]


train_path=["part-00000","part-00001"]
test_path=["part-00008"]


def main():
    total_ndcg = 0.0
    train_batch_size=5000
    test_batch_size=20000
    basedir="/Users/noelsun/Desktop/lambamart"

    train_list=[]
    test_list=[]
    for v in train_path:
        train_list.append(os.path.join(basedir,v))
    for v in test_path:
        test_list.append(os.path.join(basedir,v))

    train_data = get_data(train_list)
    test_data = get_data(test_list)


    np.random.shuffle(train_data)
    np.random.shuffle(test_data)


    iter=-1

    model = LambdaMART(None, 10, 0.005, 'sklearn')

    while iter<100:
        iter += 1
        train_start=(iter*train_batch_size)%len(train_data)
        train_end=((iter+1)*train_batch_size)%len(train_data)
        if train_end<train_start:
            continue

        test_start =(iter * test_batch_size)%(len(test_data))
        test_end = ((iter + 1) * test_batch_size)%(len(test_data))
        if test_end < test_start:
            continue

        train_batch=train_data[train_start:train_end]
        test_batch=test_data[test_start:test_end]


        model.training_data=train_batch



        model.fit()
        # model.save('./model/lambdamart_model_%d' % (iter))
        average_ndcg, predicted_scores = model.validate(test_batch, 10)

        # print(iter,model.predict(test_data[:20, 1:]))
        print(average_ndcg)



def genSamples(path_list):
    cnt=1
    except_cnt=1
    feature=[]
    key=['fea_keyword_code', 'fea_ner_code','fea_item_id', 'fea_ip0', 'fea_ip1', 'fea_ip2', 'fea_front_cate_type', 'fea_price', 'fea_6', 'fea_brand_id', 'fea_type', 'fea_sub_type', 'fea_sub_status', 'fea_physical_type', 'fea_sale_type', 'fea_15', 'fea_16']
    for path in path_list:
        print(path)
        f = open(path,"r")
        for line in f.readlines():
            cnt+=1
            try:
                jsobj=json.loads(line)
                fea_list=[int(jsobj["is_clk"][0])]
#                 print("xxx",jsobj)
                for k in key:
                    fea=list(map(lambda x:int(x),jsobj[k]))
                    if k=="fea_ip2" or k=="fea_ip1" or k=="fea_ip0":
                        if len(fea)<6:
                            fea.extend([0]*(6-len(fea)))
                        else:
                            fea=fea[:6]
                        fea_list.extend(fea)
                    elif k=="fea_ner_code":
                        if len(fea)<4:
                            fea.extend([0]*(4-len(fea)))
                        else:
                            fea=fea[:4]
                        fea_list.extend(fea)
                    else:
                        fea_list.extend(fea)
                feature.append(fea_list)
#                 print("feature",feature)
                if len(feature)%100000==0:
                    print(cnt,except_cnt)

            except:
                except_cnt+=1

    return feature



if __name__ == '__main__':
    main()