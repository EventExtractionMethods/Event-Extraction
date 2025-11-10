'''
Author: Derry
Date: 2024-04-11 05:02:03
LastEditors: Derry
Email: drlv@mail.ustc.edu.cn
LastEditTime: 2024-04-12 23:29:33
FilePath: /LLM/score1.py
Description: Coding by drlv of USTC
'''
import Levenshtein
import json
import difflib
from itertools import permutations
from tqdm import tqdm
import random


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                data.append(None)
    return data


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def string_similar1(s1, s2):
    if s1 == s2:
        return 1
    # 最长公共子序列
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def string_similar2(s1, s2):
    # 使用编辑距离度量
    return 1 - 2 * Levenshtein.distance(s1, s2) / (len(s1) + len(s2))


def calculate_metrics(correct, total_gt, total_pred):
    precision = correct / total_pred if total_pred else 0
    recall = correct / total_gt if total_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (
            precision + recall) else 0
    return precision, recall, f1


def compare_annotations(gt_annotation, pred_annotation):
    correct, total_gt, total_pred = 0, 0, 0

    if type(gt_annotation) is dict:
        total_gt += len(gt_annotation)
    if type(pred_annotation) is dict:
        total_pred += len(pred_annotation)

    if total_gt == 0 or total_pred == 0:
        return correct, total_gt, total_pred

    if type(gt_annotation) is dict and type(pred_annotation) is dict:
        total_keys = list(set(gt_annotation.keys() | pred_annotation.keys()))
        # if 'mention' in total_keys:
        #     total_keys.remove('mention')
        for key in total_keys:
            if key in gt_annotation and key in pred_annotation:
                cur_gt = gt_annotation[key]
                cur_pred = pred_annotation[key]
                if type(cur_gt) is list:
                    cur_gt = ''.join(cur_gt)
                if type(cur_pred) is list:
                    cur_pred = ''.join(cur_pred)
                try:
                    if string_similar1(cur_gt, cur_pred) >= 0.6 or string_similar2(cur_gt, cur_pred) >= 0.6:
                        correct += 1
                except:
                    pass

    return correct, total_gt, total_pred



def best_match(gt_annotations, pred_annotations):
    ret_correct, ret_gt_count, ret_pred_count = 0, 0, 0

    gt_tmp = []
    pred_tmp = []
    #print(pred_annotations)
    for i in range(len(gt_annotations)):
        gt_tmp.append(gt_annotations[i])

    for i in range(len(pred_annotations)):
        pred_tmp.append(pred_annotations[i])
    while True:
        if len(gt_tmp) == 0 or len(pred_tmp) == 0:
            break
        cur_corrent = -1
        cur_gt_count = 0
        cur_pred_count = 0

        if len(gt_tmp) > len(pred_tmp):
            cur_pred_index = 0
            for i in range(len(gt_tmp)):
                cur_gt = gt_tmp[i]
                cur_pred = pred_tmp[cur_pred_index]
                correct, gt_count, pred_count = compare_annotations(
                    cur_gt, cur_pred)
                if correct > cur_corrent:
                    cur_gt_index = i
                    cur_corrent = correct
                    cur_gt_count = gt_count
                    cur_pred_count = pred_count
        else:
            cur_gt_index = 0
            for i in range(len(pred_tmp)):
                cur_gt = gt_tmp[cur_gt_index]
                cur_pred = pred_tmp[i]
                correct, gt_count, pred_count = compare_annotations(
                    cur_gt, cur_pred)
                if correct > cur_corrent:
                    cur_pred_index = i
                    cur_corrent = correct
                    cur_gt_count = gt_count
                    cur_pred_count = pred_count

        ret_correct += cur_corrent
        ret_gt_count += cur_gt_count
        ret_pred_count += cur_pred_count
        # print("############################################")
        # print(cur_gt_index)
        # print(cur_pred_index)
        gt_tmp.pop(cur_gt_index)
        pred_tmp.pop(cur_pred_index)

    for i in range(len(pred_tmp)):
        correct, gt_count, pred_count = compare_annotations(None, pred_tmp[i])
        ret_correct += correct
        ret_gt_count += gt_count
        ret_pred_count += pred_count

    for i in range(len(gt_tmp)):
        correct, gt_count, pred_count = compare_annotations(gt_tmp[i], None)
        ret_correct += correct
        ret_gt_count += gt_count
        ret_pred_count += pred_count

    return ret_correct, ret_gt_count, ret_pred_count



def evaluate(gt_file, pred_file):
    gt_data = load_jsonl(gt_file)
    pred_data = load_jsonl(pred_file)
    total_correct, total_gt, total_pred = 0, 0, 0

    for i in range(len(gt_data)):
        print(i)
        cur_gt = gt_data[i]
        cur_pred = pred_data[i]
        if cur_gt == None:
            cur_gt=[]
        if cur_pred == None:
            cur_pred=[]
        correct, gt_num, pred_num = best_match(
            cur_gt, cur_pred)
        total_correct += correct
        total_gt += gt_num
        total_pred += pred_num

    print(total_correct, total_gt, total_pred)
    precision, recall, f1 = calculate_metrics(
        total_correct, total_gt, total_pred)
    return precision, recall, f1



# todo = range(226, 236)  # [71,72,73,74,75,76,77,78,79,80]
#
# for i in todo:
#     precision, recall, f1 = evaluate(f'./data/p0_gt.jsonl', f'./output/{i}.jsonl')
#     print(f'Precision={100 * precision:.2f}%, Recall={100 * recall:.2f}%, F1={100 * f1:.2f}%')

precision, recall, f1 = evaluate('./data/p0_gtns.jsonl', './output/p0_qwen_14.jsonl')
print(f'Precision={100*precision:.2f}%, Recall={100*recall:.2f}%, F1={100*f1:.2f}%')
