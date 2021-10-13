import os
import pandas as pd
from sklearn import metrics

# debug
from IPython.display import display

def make_pred_df(wav_names, section, domain, label, pred):
    """[summary]

    Args:
        wav_names (list): [description]
        section (numpy.array): [description]
        domain (numpy.array): [description]
        label (numpy.array): [description]
        pred (numpy.array): [description]

    Returns:
        pandas.DataFrame: [description]
    """
    wav_names = [os.path.basename(wav_name) for wav_name in wav_names]
    pred_df = pd.DataFrame({
        'wav_name': wav_names,
        'section': section,
        'domain': domain,
        'label': label,
        'pred': pred,
        })
    return pred_df

### metric ###
def calc_auc(y_true, y_pred, max_fpr=0.1):
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
    return auc, p_auc

def calc_dcase2021_task2_score(pred_df, prefix='Source'):
    score_list = []
    uniq_section = pred_df['section'].unique()
    for sec in uniq_section:
        sec_pred = pred_df[pred_df['section'] == sec]
        auc, p_auc = calc_auc(sec_pred['label'], sec_pred['pred'])
        # [[auc, p_auc],,, ]
        score_list.append([auc, p_auc])
    # [[auc, p_auc], [auc, p_auc], [auc, p_auc]] -> df
    score_df = pd.DataFrame(score_list, columns = ['AUC', 'pAUC'])
    # rename
    score_df = score_df.rename(index=lambda num: f'{prefix}_{num}')
    return score_df
   
##############