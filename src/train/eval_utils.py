from typing import Dict, Any, Optional, Union, Tuple
import re
import json
import warnings
import torch
import transformers
import pandas as pd
from transformers import ProcessorMixin
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
warnings.filterwarnings('ignore')


def preprocess_logits_for_metrics(
    logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预处理logits和labels，logits和labels取序列第一个非pad token的值

    Args:
        logits: 模型输出的logits [batch_size, seq_len, vocab_size]
        labels: 目标标签  [batch_size, seq_len]
        pad_token_id: 填充token的ID

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 处理后的logits 和 labels shape: [batch_size]
    """
    # 移动以对齐预测和标签
    shifted_logits = logits[:, :-1, :]  # [B, L-1, V]
    shifted_labels = labels[:, 1:]      # [B, L-1]

    # 获取预测值（最大概率对应的token）
    preds = shifted_logits.argmax(dim=-1)  # [B, L-1]

    # 找到每个样本中第一个非pad的位置索引
    non_pad_mask = (shifted_labels != pad_token_id)
    first_non_pad_idx = non_pad_mask.float().cumsum(dim=1).eq(1)  # [B, L-1]

    # 从每行中提取第一个非pad位置的预测和标签
    first_preds = preds[first_non_pad_idx]         # [B]
    first_labels = shifted_labels[first_non_pad_idx]  # [B]

    return first_preds, first_labels



class MetricsCalculator:
    """评估指标计算器，支持批量计算和累积统计"""

    def __init__(self, processor: ProcessorMixin, predict_with_generate: bool = False):
        self.processor = processor
        self.predict_with_generate = predict_with_generate
        self.reset_stats()

    def reset_stats(self):
        """重置统计数据"""
        self.total_correct = 0
        self.total_samples = 0
        self.pred_ans_list = []

    @staticmethod
    def extract_ans(ans):
        pattern = re.compile(r"The answer is \(([A-Z])\).")
        res = pattern.findall(ans)
    
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED"
        return answer

    @torch.no_grad()
    def __call__(
        self,
        eval_preds: transformers.trainer_utils.EvalPrediction,
        compute_result: bool = False,
    ) -> Dict[str, float]:
        """计算批次或完整评估的指标

        Args:
            eval_preds: 评估预测结果
            compute_result: 是否计算最终结果

        Returns:
            Dict[str, float]: 评估指标字典
        """

        # 处理当前批次
        if self.predict_with_generate:
            preds, targets = eval_preds
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds, targets = preprocess_logits_for_metrics(preds, targets)

        # 解码预测和目标
        decoded_preds = self.processor.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        decoded_targets = self.processor.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # 计算正确预测数
        batch_correct = 0
        assert len(decoded_preds) == len(decoded_targets)

        for pred, target in zip(decoded_preds, decoded_targets):
            if pred == target:
                batch_correct += 1
            self.pred_ans_list.append([pred, target])

        # 累积统计
        self.total_correct += batch_correct
        self.total_samples += len(decoded_preds)
        # 如果是最终计算
        if compute_result:
            accuracy = (
                self.total_correct / self.total_samples if self.total_samples > 0 else 0
            )
            with open("pred_ans.json", "w") as f:
                json.dump(self.pred_ans_list, f, indent=4)
            
            self.reset_stats()
            return {"accuracy": accuracy}



########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    weights = (1. / gram,) * gram

    return sentence_bleu([reference_tokens], hypothesis_tokens, weights)


def caculate_bleu(results, data, gram):
    bleus = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()
        if target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)

    avg_bleu = sum(bleus) / len(bleus)

    return avg_bleu


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def caculate_rouge(results, data):
    rouges = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()
        if prediction == "":
            continue
        if target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)

    avg_rouge = sum(rouges) / len(rouges)
    return avg_rouge


########################
## Sentence Similarity
########################
def similariry_score(str1, str2, model):
    # compute embedding for both lists
    embedding_1 = model.encode(str1, convert_to_tensor=True)
    embedding_2 = model.encode(str2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    return score


def caculate_similariry(results, data, model):
    scores = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()

        score = similariry_score(target, prediction, model)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score


def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    if len(total_pd) == 0:
        acc = 0.0
    else:
        acc = "{:.5f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(pred_ans_data, rationale_data, results_reference, data_file):
    num = len(pred_ans_data)

    # read data file
    sqa_data = json.load(open(data_file))

    # construct pandas data
    res_pd = pd.DataFrame(sqa_data)

    # update data
    for index, row in res_pd.iterrows():

        # res_pd.loc[index, 'no_context'] = True if (not row['conversations'] and not row['image']) else False
        # res_pd.loc[index, 'has_text'] = True if row['conversations'] else False
        # res_pd.loc[index, 'has_image'] = True if row['image'] else False
        # res_pd.loc[index, 'has_text_image'] = True if (row['conversations'] and row['image']) else False

        label = row['answer'].strip()
        pred = pred_ans_data[res_pd.loc[index, 'id']]
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100

    # rationale quality
    ## BLEU
    bleu1 = caculate_bleu(rationale_data, results_reference, gram=1)
    bleu4 = caculate_bleu(rationale_data, results_reference, gram=4)

    ## Rouge-L
    rouge = caculate_rouge(rationale_data, results_reference)

    ## Similarity
    model = SentenceTransformer('all-MiniLM-L6-v2').cuda()
    similariry = caculate_similariry(rationale_data, results_reference, model)

    scores = {
            "answer":{
                'acc_natural':
                get_acc_with_contion(res_pd, 'subject', 'natural science'),
                'acc_social':
                get_acc_with_contion(res_pd, 'subject', 'social science'),
                'acc_language':
                get_acc_with_contion(res_pd, 'subject', 'language science'),
                'acc_has_text':
                get_acc_with_contion(res_pd, 'has_text', True),
                'acc_has_image':
                get_acc_with_contion(res_pd, 'has_image', True),
                'acc_no_context':
                get_acc_with_contion(res_pd, 'no_context', True),
                'acc_grade_1_6':
                get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
                'acc_grade_7_12':
                get_acc_with_contion(res_pd, 'grade', values=['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
                'acc_average':
                "{:.5f}".format(acc_average),
            },
            "rationale":{
                'bleu1': bleu1 * 100,
                'bleu4': bleu4 * 100,
                'rouge': rouge * 100,
                'similariry': similariry * 100,
            }
    }
    return scores
