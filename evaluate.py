import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import spearmanr
import numpy as np

# ✅ 安全计算 Spearman，避免全常数导致 NaN
def safe_spearmanr(a, b):
    if len(set(a)) <= 1 or len(set(b)) <= 1:
        return np.nan
    return spearmanr(a, b).correlation

def evaluate_prediction(true_path='true.csv', pred_path='result.csv'):
    # 读取预测与真实结果
    df_true = pd.read_csv(true_path, dtype=str, encoding='utf-8')
    df_pred = pd.read_csv(pred_path, dtype=str, encoding='utf-8')

    # 确保只有两列：涨幅最大、涨幅最小
    cols = ['涨幅最大股票代码', '涨幅最小股票代码']
    df_true = df_true[cols].astype(str)
    df_pred = df_pred[cols].astype(str)

    results = {}

    for col in cols:
        true_list = df_true[col].tolist()
        pred_list = df_pred[col].tolist()

        # 命中统计
        true_set = set(true_list)
        pred_set = set(pred_list)
        hit_set = true_set & pred_set

        print(f"\n✅ 命中股票（{col}）数量：{len(hit_set)}")
        print(f"🎯 命中股票列表：{sorted(hit_set) if hit_set else '无'}")

        # 二分类指标计算
        all_items = list(true_set.union(pred_set))
        y_true = [1 if stock in true_set else 0 for stock in all_items]
        y_pred = [1 if stock in pred_set else 0 for stock in all_items]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 安全 Spearman
        spearman_score = safe_spearmanr(
            [true_list.index(x) if x in true_list else len(true_list) for x in pred_list],
            list(range(len(pred_list)))
        )

        results[col] = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Spearman': spearman_score
        }

    # 平均指标
    avg_f1 = (results[cols[0]]['F1'] + results[cols[1]]['F1']) / 2
    avg_spearman = (results[cols[0]]['Spearman'] + results[cols[1]]['Spearman']) / 2

    # 加权评分
    final_score = 0.6 * avg_f1 + 0.4 * np.nan_to_num(avg_spearman)

    # 输出指标
    print("\n📊 预测结果评估指标：")
    for col in cols:
        print(f"\n【{col}】")
        for k, v in results[col].items():
            print(f"{k:<10}: {v:.4f}")
    print("\n✅ 加权综合评分 Final Score: {:.4f}".format(final_score))

if __name__ == "__main__":
    evaluate_prediction('true.csv', 'result.csv')
