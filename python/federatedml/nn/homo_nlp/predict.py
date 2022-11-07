from federatedml.nn.homo_nlp._torch import FedLightModule

import json
import pandas as pd
from ark_nlp.model.tc.bert import Dataset as BertDataset

from ark_nlp.model.tc.bert import Predictor
from ark_nlp.model.tc.bert import Tokenizer
import os
import time
import csv


def predict_nlp(model_path, data_path, result_path, metric_path, istest = "test"):
    '''
    istest: test or predict
    '''
    # model = FedLightModule.load_from_checkpoint('/data/projects/fate/examples/mywork/cervical/code/model.ckpt')
    basepath = '/data/projects/fate/examples/results'
    cat2id = {'其他': 0,
        '功效作用': 1,
        '医疗费用': 2,
        '后果表述': 3,
        '就医建议': 4,
        '指标解读': 5,
        '治疗方案': 6,
        '注意事项': 7,
        '疾病表述': 8,
        '病因分析': 9,
        '病情诊断': 10}

    model = FedLightModule.load_from_checkpoint(model_path)
    tokenizer = Tokenizer(vocab='bert-base-chinese', max_seq_len=50)

    tc_predictor_instance = Predictor(model.model, tokenizer, cat2id)
    # test_df = pd.read_json('/home/qi/sunhaifeng/text_classification/KUAKE-QIC_dev.json')
    test_df = pd.read_json(data_path)

    true_num = 0
    rowlist = []
    if istest == "predict":
        for id_, text_ in zip(test_df['id'], test_df['query']):
            predict_ = tc_predictor_instance.predict_one_sample(text_)[0] 
            predict_instance = [id_,text_,predict_]
            rowlist.append(predict_instance)
    else:
        for id_, text_, ture_label_ in zip(test_df['id'], test_df['query'], test_df['label']):
            predict_ = tc_predictor_instance.predict_one_sample(text_)[0]
            predict_instance = [id_, text_, ture_label_, predict_]
            rowlist.append(predict_instance)
            if ture_label_ == predict_:
                true_num += 1
        metrics = f"Test Accuracy:{100*(true_num/len(rowlist)):>0.2f}%"

    if istest == "test":
        output_f_metric = open(metric_path,"a",encoding = 'utf-8')
        csv_f_m = csv.writer(output_f_metric)
        csv_f_m.writerow([f'Test Accuracy',f'{100*(true_num/len(rowlist)):>0.2f}%'])

    
    # output_path = os.path.join(basepath,jobid + "_" + str(int(time.time())) + ".csv")
    output_f = open(result_path,'a',newline='',encoding="utf-8-sig")
    csv_f = csv.writer(output_f)
    header = ['id','query','predict label'] if istest == "predict" else  ['id','query','true label','predict label'] 
    csv_f.writerow(header)
    csv_f.writerows(rowlist)


def predict_nlp_one_sample(model_path, data):
    '''
    istest: test or predict
    '''
    cat2id = {'其他': 0,
        '功效作用': 1,
        '医疗费用': 2,
        '后果表述': 3,
        '就医建议': 4,
        '指标解读': 5,
        '治疗方案': 6,
        '注意事项': 7,
        '疾病表述': 8,
        '病因分析': 9,
        '病情诊断': 10}

    model = FedLightModule.load_from_checkpoint(model_path)
    tokenizer = Tokenizer(vocab='bert-base-chinese', max_seq_len=50)

    tc_predictor_instance = Predictor(model.model, tokenizer, cat2id)
    predict_ = tc_predictor_instance.predict_one_sample(data)[0] 
    return predict_



if __name__ == "__main__" :
    # predict_nlp(model_path = '/data/projects/fate/fateflow/jobs/202210130408229032090/guest/9997/homo_nlp_0/202210130408229032090_homo_nlp_0/0/task_executor/b1a7b7684aac11edaca90242c0a70064/model.ckpt',
    # data_path = '/data/projects/fate/examples/hfwork/KUAKE-QIC_dev.json',result_path = "./result.csv",metric_path = "./metric.csv",istest = "predict")
    p = predict_nlp_one_sample(model_path = 'fateflow/jobs/202210201957297490170/host/9997/homo_nlp_0/202210201957297490170_homo_nlp_0/0/task_executor/64b5b73a506e11ed86d10242c0a70064/model.ckpt',
    data = "老是拉肚子怎么办")
    print(p)
