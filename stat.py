import os
import pickle
import csv
import torch
import numpy as np

lm_file_path = os.path.join("output/hotpot/base/meta-llama/Meta-Llama-3-8B", "lm.pkl")
knn_file_path = os.path.join("output/hotpot/math_10_2048_14/meta-llama/Meta-Llama-3-8B", "knn.pkl")

lm_data_list = []
knn_data_list = []

with open(lm_file_path, "rb") as f:
    while True:
        try:
            data = pickle.load(f)
            lm_data_list.extend(data)  # Extend the list if data is also a list of dictionaries
        except EOFError:
            break
print("read lm file")

with open(knn_file_path, "rb") as f:
    while True:
        try:
            data = pickle.load(f)
            knn_data_list.extend(data)  # Extend the list if data is also a list of dictionaries
        except EOFError:
            break
print("read knn file")

assert len(lm_data_list) == len(knn_data_list), "The length of lm_data_list and knn_data_list must be the same"

lm_correct_knn_wrong = 0
lm_wrong_knn_correct = 0
label_diff_count = 0

csv_header = [
    "question", "answers",
    "lm_prediction", "knn_prediction",
    "lm_loss", "knn_loss", "loss_diff",
    "lm_f1", "knn_f1",
    "average", "max", "min", "variance"
]

csv_file_path = os.path.join("output/hotpot/math_10_2048_14/meta-llama/Meta-Llama-3-8B", "hotpot_results.csv")
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

    for lm_data, knn_data in zip(lm_data_list, knn_data_list):
        question = lm_data['question']
        answers = lm_data['answers']
        lm_pred_label = lm_data['prediction']
        knn_pred_label = knn_data['prediction']
        lm_loss = lm_data['loss']
        knn_loss = knn_data['loss']
        lm_is_correct = lm_data['correct']
        knn_is_correct = knn_data['correct']
        lm_f1 = lm_data['F1']
        knn_f1 = knn_data['F1']

        if lm_pred_label != knn_pred_label:
            label_diff_count += 1
        if lm_is_correct and not knn_is_correct:
            lm_correct_knn_wrong += 1
        elif not lm_is_correct and knn_is_correct:
            lm_wrong_knn_correct += 1
        elif not lm_is_correct and not knn_is_correct:
            if knn_f1 > lm_f1:
                lm_wrong_knn_correct += 1
            elif knn_f1 < lm_f1:
                lm_correct_knn_wrong += 1

        lm_logits = torch.tensor(lm_data['logits'])
        knn_logits = torch.tensor(knn_data['logits'])

        
        # 计算统计量
        diff_average, diff_max, diff_min, diff_variance = None, None, None, None

        if lm_logits.size() == knn_logits.size():
            logits_diff = knn_logits - lm_logits
            # 计算统计量
            diff_average = logits_diff.mean(axis=-1).mean().item()
            diff_max = logits_diff.max(axis=-1).values.mean().item()
            diff_min = logits_diff.min(axis=-1).values.mean().item()
            diff_variance = logits_diff.var(axis=-1).mean().item()

        loss_diff = knn_loss - lm_loss

        writer.writerow([
            question, answers,
            lm_pred_label, knn_pred_label,
            lm_loss, knn_loss, loss_diff,
            lm_f1, knn_f1,
            diff_average, diff_max, diff_min, diff_variance
        ])

total_samples = len(lm_data_list)
ratio_label_diff = label_diff_count / total_samples
ratio_lm_correct_knn_wrong = lm_correct_knn_wrong / label_diff_count
ratio_lm_wrong_knn_correct = lm_wrong_knn_correct / label_diff_count

print(f"LM correct, KNN wrong: {lm_correct_knn_wrong}")
print(f"LM wrong, KNN correct: {lm_wrong_knn_correct}")
print(f"LM and KNN labels different: {label_diff_count}")
print(f"Change Ratio (LM and KNN labels different): {ratio_label_diff}")
print(f"Negative Ratio (LM correct, KNN wrong): {ratio_lm_correct_knn_wrong}")
print(f"Positive Ratio (LM wrong, KNN correct): {ratio_lm_wrong_knn_correct}")
print(f"Neutral Ratio (LM wrong, KNN correct): {1 - ratio_lm_wrong_knn_correct - ratio_lm_correct_knn_wrong}")
