# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import copy
import gc


def read_feature(filename):
    feature_names = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            feature_names.append(line.replace("\n", ""))
    print feature_names
    return feature_names


'''----------------------------------------------数据预处理-----------------------------------------------------------'''
data_dir = "./home-credit-default-risk/"
num_iterations = 1000
lgb_param = {'application':'binary', 'learning_rate':0.02, 'num_leaves':24, 'feature_fraction':1,
                 'bagging_fraction':1, 'lambda_l1':0.1, 'lambda_l2':0.1, 'min_split_gain':0.01, 'is_unbalance':True,
                 'metric':'auc'}
lgb_param_reg = {'application':'regression', 'learning_rate':0.01, 'num_leaves':24, 'feature_fraction':1,
                 'bagging_fraction':1, 'lambda_l1':0.1, 'lambda_l2':0.1, 'min_split_gain':0.01}
# bureau = pd.read_csv(data_dir + "bureau.csv")
# bureau_balance = pd.read_csv(data_dir + "bureau_balance.csv")
# credit_card_balance = pd.read_csv(data_dir + "credit_card_balance.csv")
# installments_payments = pd.read_csv(data_dir + "installments_payments.csv")
# pos_cash_balance = pd.read_csv(data_dir + "POS_CASH_balance.csv")
# previous_application = pd.read_csv(data_dir + "previous_application.csv")


def helper_stat(data, item_name, suffix=""):
    tmp = data.copy()[[item_name, "SK_ID_CURR"]].groupby(["SK_ID_CURR"])
    tmp_max = tmp.max()
    tmp_max.columns = [item_name + "_MAX" + suffix]
    tmp_min = tmp.min()
    tmp_min.columns = [item_name + "_MIN" + suffix]
    tmp_mean = tmp.mean()
    tmp_mean.columns = [item_name + "_MEAN" + suffix]
    tmp_std = tmp.std()
    tmp_std.columns = [item_name + "_STD" + suffix]
    tmp_median = tmp.median()
    tmp_median.columns = [item_name + "_MEDIAN" + suffix]
    tmp_sum = tmp.sum()
    tmp_sum.columns = [item_name + "_SUM" + suffix]
    # wm = lambda x: np.average(x, weights=-1.0/(1 + np.log1p(np.abs(data.loc[x.index, "DAYS_CREDIT_UPDATE"]))))
    # tmp_wavg = tmp.agg({item_name: wm})
    # tmp_wavg.columns = [item_name + "_WAVG" + suffix]
    # return tmp_max.join(tmp_min, how="left").join(tmp_mean, how="left").join(tmp_std,
    #         how="left").join(tmp_median, how="left").join(tmp_sum, how="left").join(tmp_wavg, how="left")
    return tmp_max.join(tmp_min, how="left").join(tmp_mean, how="left").join(tmp_std,
                how="left").join(tmp_median, how="left").join(tmp_sum, how="left")


# 对application_train做预处理
# EXT_SOURCE_1，EXT_SOURCE_2，EXT_SOURCE_3空值处理办法：保留空值，因为这个字段有区分能力，但是含义不直接，不好直接填均值或0
# 必须填空值时使用均值
def proprocess_application_train_test(data, is_output=False, file_name=None):
    # AMT_ANNUITY空值处理办法：用其余所有人的 AMT_CREDIT／AMT_ANNUITY 的均值近似为这些用户需要还的期数，用 AMT_CREDIT 除以这个值
    ratio_avg = np.mean(data["AMT_CREDIT"] * 1.0 / data["AMT_ANNUITY"])
    unnull_part = data[~data["AMT_ANNUITY"].isnull()].copy()
    null_part = data[data["AMT_ANNUITY"].isnull()].copy()
    null_part["AMT_ANNUITY"] = null_part["AMT_CREDIT"] / ratio_avg
    data = unnull_part.append(null_part)

    # AMT_GOODS_PRICE空值处理办法：用AMT_CREDIT列代替
    unnull_part = data[~data["AMT_GOODS_PRICE"].isnull()].copy()
    null_part = data[data["AMT_GOODS_PRICE"].isnull()].copy()
    null_part["AMT_GOODS_PRICE"] = null_part["AMT_CREDIT"].copy()
    data = unnull_part.append(null_part)

    # NAME_TYPE_SUITE空值处理办法：用Unaccompanied填充
    unnull_part = data[~data["NAME_TYPE_SUITE"].isnull()].copy()
    null_part = data[data["NAME_TYPE_SUITE"].isnull()].copy()
    null_part["NAME_TYPE_SUITE"] = "Unaccompanied"
    data = unnull_part.append(null_part)

    # OWN_CAR_AGE空值处理办法：空值里有两个记录是有车的，其余均没车, 空值记录里填-1（为了训练集和测试集同步，而且测试集不会少数据）
    unnull_part = data[~data["OWN_CAR_AGE"].isnull()].copy()
    null_part = data[data["OWN_CAR_AGE"].isnull()].copy()
    null_part["OWN_CAR_AGE"] = -1
    data = unnull_part.append(null_part)

    # OCCUPATION_TYPE空值处理办法: 作为另外一种职业传入，无职业者的欠款率明显低于平均水平，而且收入分布于整体的收入分布类似
    unnull_part = data[~data["OCCUPATION_TYPE"].isnull()].copy()
    null_part = data[data["OCCUPATION_TYPE"].isnull()].copy()
    null_part["OCCUPATION_TYPE"] = "third_party"
    data = unnull_part.append(null_part)

    # CNT_FAM_MEMBERS空值处理办法：就2个空值，用众数吧
    unnull_part = data[~data["CNT_FAM_MEMBERS"].isnull()].copy()
    null_part = data[data["CNT_FAM_MEMBERS"].isnull()].copy()
    null_part["CNT_FAM_MEMBERS"] = unnull_part[["CNT_FAM_MEMBERS"]].mode().loc[0, "CNT_FAM_MEMBERS"]
    data = unnull_part.append(null_part)

    # APARTMENTS_AVG空值处理办法：因为这个记录的是客户居住地的信息，和客户是否有房子没关系，猜想信息太细大部分客户无法提供
    # 所以对这类字段，AVG后缀的字段用非空值的均值填充，MODE后缀的字段用非空值的众数填充，MIDI后缀的字段用中位数填充

    def helper(names, data_static, dataframe):
        data_tmp =  dataframe.copy()
        for name in names:
            unnull_part = data_tmp[~data_tmp[name].isnull()].copy()
            null_part = data_tmp[data_tmp[name].isnull()].copy()
            null_part[name] = data_static[name]
            data_tmp = unnull_part.append(null_part)
        return data_tmp

    avg_names = ["APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
                 "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG",
                 "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
                 "NONLIVINGAREA_AVG"]
    data_avg = data[avg_names].mean()
    data = helper(avg_names, data_avg, data)
    mode_names = ["APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE",
                  "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE",
                  "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE",
                  "NONLIVINGAREA_MODE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE", "WALLSMATERIAL_MODE",
                  "EMERGENCYSTATE_MODE"]
    data_mode = data[mode_names].mode().loc[0, :]
    data = helper(mode_names, data_mode, data)
    medi_names = ["APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI",
                  "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI",
                  "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI",
                  "NONLIVINGAREA_MEDI"]
    data_median = data[medi_names].median()
    data = helper(medi_names, data_median, data)

    # OBS_30_CNT_SOCIAL_CIRCLE,DEF_30_CNT_SOCIAL_CIRCLE,OBS_60_CNT_SOCIAL_CIRCLE,DEF_60_CNT_SOCIAL_CIRCLE空值处理办法：
    # 全部填0，因为这几个字段以0居多（至少超过50%以上），而且空值样本数量有限
    # AMT_REQ_CREDIT_BUREAU_HOUR,AMT_REQ_CREDIT_BUREAU_DAY,AMT_REQ_CREDIT_BUREAU_WEEK,
    # AMT_REQ_CREDIT_BUREAU_MON,AMT_REQ_CREDIT_BUREAU_QRT,AMT_REQ_CREDIT_BUREAU_YEAR空值处理办法：
    # 前五个字段填0，最后一个字段用均值
    zero_names = ["OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
                  "DEF_60_CNT_SOCIAL_CIRCLE", "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
                  "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT"]
    for name in zero_names:
        unnull_part = data[~data[name].isnull()].copy()
        null_part = data[data[name].isnull()].copy()
        null_part[name] = 0
        data = unnull_part.append(null_part)
    unnull_part = data[~data["AMT_REQ_CREDIT_BUREAU_YEAR"].isnull()].copy()
    null_part = data[data["AMT_REQ_CREDIT_BUREAU_YEAR"].isnull()].copy()
    null_part["AMT_REQ_CREDIT_BUREAU_YEAR"] = np.mean(unnull_part["AMT_REQ_CREDIT_BUREAU_YEAR"])
    data = unnull_part.append(null_part)

    # DAYS_EMPLOYED大于0说明没工作，尝试将所有大于0的数据转成0
    data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].apply(lambda x: 0 if (x > 0) else x)

    if is_output:
        data.to_csv("home-credit-default-risk-preprocess/" + file_name, index=None)
    return data


# 这个预处理的过程就是计算特征的过程
def preprocess_burau(data, is_output=False, file_name=None):
    # 计算该application下，所有的credit记录里CREDIT_ACTIVE字段各种类型的占比：
    tmp = pd.get_dummies(data[["SK_ID_CURR", "CREDIT_ACTIVE"]].copy())
    result = tmp.groupby(["SK_ID_CURR"]).mean()

    # CREDIT_CURRENCY 做onehot，聚合后做加法
    tmp = pd.get_dummies(data[["SK_ID_CURR", "CREDIT_CURRENCY"]].copy())
    tmp = tmp.groupby(["SK_ID_CURR"]).sum()
    result = result.join(tmp, how="left")

    # CREDIT_DAY_OVERDUE：该credit记录从逾期开始到当前application一共经过了多少天。可以从非0均值，最大值，总数三个角度衡量用户的信用情况
    tmp = data[["SK_ID_CURR", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE"]].copy()
    tmp_sum = tmp.groupby(["SK_ID_CURR"]).sum()
    tmp_sum.columns = ["CREDIT_DAY_OVERDUE_SUM", "AMT_CREDIT_MAX_OVERDUE_SUM"]
    tmp_max = tmp.groupby(["SK_ID_CURR"]).max()
    tmp_max.columns = ["CREDIT_DAY_OVERDUE_MAX", "AMT_CREDIT_MAX_OVERDUE_MAX"]
    tmp_mean = tmp[tmp["CREDIT_DAY_OVERDUE"] > 0][["SK_ID_CURR", "CREDIT_DAY_OVERDUE"]].groupby(["SK_ID_CURR"]).mean()
    tmp_mean.columns = ["CREDIT_DAY_OVERDUE_MEAN"]
    # 因为是非0均值，所以得一个字段一个字段的计算
    tmp_mean_2 = tmp[tmp["AMT_CREDIT_MAX_OVERDUE"] > 0][["SK_ID_CURR", "AMT_CREDIT_MAX_OVERDUE"]].groupby(["SK_ID_CURR"]).mean()
    tmp_mean_2.columns = ["AMT_CREDIT_MAX_OVERDUE_MEAN"]
    tmp = tmp_sum.join(tmp_max, how="left").join(tmp_mean, how="left").join(tmp_mean_2, how="left").fillna(0)
    result = result.join(tmp, how="left")

    # AMT_CREDIT_MAX_OVERDUE：最大欠款金额，反映用户在规定还款期内偿还了多少贷款，可以贷款金额先做差值（比值），再计算最大值（均值）
    # 还有用户的历史欠款率
    tmp = data[["SK_ID_CURR", "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM"]].copy().dropna()
    tmp["AMT_CREDIT_IS_OVERDUE"] = tmp["AMT_CREDIT_MAX_OVERDUE"].apply(lambda x: 1 if (x > 0) else 0)
    tmp["CREDIT_MINUX_OVERDUE"] = (tmp["AMT_CREDIT_SUM"] - tmp["AMT_CREDIT_MAX_OVERDUE"]).apply(lambda x: x if (x >= 0) else 0)
    tmp["CREDIT_DIVIDE_OVERDUE"] = (tmp["AMT_CREDIT_SUM"] * 1.0 / tmp["AMT_CREDIT_MAX_OVERDUE"]).apply(lambda x: x if (x >= 0) else 0)
    tmp_max = tmp[["SK_ID_CURR", "CREDIT_MINUX_OVERDUE", "CREDIT_DIVIDE_OVERDUE"]].groupby(["SK_ID_CURR"]).max()
    tmp_max.columns = ["CREDIT_MINUX_OVERDUE_MAX", "CREDIT_DIVIDE_OVERDUE_MAX"]
    tmp_mean = tmp[["SK_ID_CURR", "CREDIT_MINUX_OVERDUE", "CREDIT_DIVIDE_OVERDUE", "AMT_CREDIT_IS_OVERDUE"]].groupby(["SK_ID_CURR"]).mean()
    tmp_mean.columns = ["CREDIT_MINUX_OVERDUE_MEAN", "CREDIT_DIVIDE_OVERDUE_MEAN", "CREDIT_OVERDUE_RATIO"]
    tmp = tmp_max.join(tmp_mean, how="left")
    result = result.join(tmp, how="left")

    helper = helper_stat

    # CNT_CREDIT_PROLONG：信贷局给出该credit报告，延期了多少次。该字段不知道详细含义，只能进行不断的尝试
    tmp = data[["SK_ID_CURR", "CNT_CREDIT_PROLONG", "DAYS_CREDIT_UPDATE"]].copy()
    tmp = helper(tmp, "CNT_CREDIT_PROLONG")
    result = result.join(tmp, how="left")
    del tmp
    gc.collect()

    # 当前贷款金额，最大值，均值，总和，中位数可以反映用户的贷款水平
    tmp = data[["AMT_CREDIT_SUM", "SK_ID_CURR", "DAYS_CREDIT_UPDATE"]].copy()
    tmp = helper(tmp.dropna(), "AMT_CREDIT_SUM")
    result = result.join(tmp, how="left")
    del tmp
    tmp = data[data["CREDIT_ACTIVE"] == "Active"][["SK_ID_CURR", "AMT_CREDIT_SUM", "DAYS_CREDIT_UPDATE"]].copy()
    tmp = helper(tmp.dropna(), "AMT_CREDIT_SUM", "_ACTIVE")
    result = result.join(tmp, how="left")
    del tmp
    tmp = data[data["CREDIT_ACTIVE"] == "Closed"][["SK_ID_CURR", "AMT_CREDIT_SUM", "DAYS_CREDIT_UPDATE"]].copy()
    tmp = helper(tmp.dropna(), "AMT_CREDIT_SUM", "_CLOSED")
    result = result.join(tmp, how="left")
    del tmp
    gc.collect()

    # AMT_CREDIT_SUM_DEBT：当前的欠款金额，反映当前用户的欠款情况，不仅可以参考CNT_CREDIT_PROLONG构造特征。还能对所有历史欠款金额求和，
    # 表示用户当前已有的债务负担，再进一步可以和当前application贷款金额做对比（差值或者比值）
    tmp = data[["SK_ID_CURR", "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT_UPDATE"]].copy()
    tmp = helper(tmp.dropna(), "AMT_CREDIT_SUM_DEBT")
    result = result.join(tmp, how="left")
    del tmp
    gc.collect()

    # AMT_CREDIT_SUM_LIMIT：计算总和，最大值，最小值，均值，方差，中位数等，因为额度的多少代表了用户在那个借贷结构的信用水平
    tmp = data[["SK_ID_CURR", "AMT_CREDIT_SUM_LIMIT", "DAYS_CREDIT_UPDATE"]].copy()
    tmp = helper(tmp.dropna(), "AMT_CREDIT_SUM_LIMIT")
    result = result.join(tmp, how="left")
    del tmp
    gc.collect()

    # AMT_CREDIT_SUM_OVERDUE：当前已经逾期的金额，这部分金额不仅能代表用户的负担，而且增加了逾期的含义，可以和总欠款金额做对比（差值或者比值）
    tmp = data[["SK_ID_CURR", "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT_UPDATE"]].copy()
    tmp = helper(tmp.dropna(), "AMT_CREDIT_SUM_OVERDUE")
    result = result.join(tmp, how="left")
    del tmp
    gc.collect()

    # NUMBER OF PAST LOANS PER CUSTOMER
    tmp = data[["SK_ID_CURR", "SK_ID_BUREAU"]].copy()
    tmp = tmp.groupby(["SK_ID_CURR"]).count()
    tmp.columns = ["HISTORY_CREDIT_COUNT"]
    result = result.join(tmp, how="left")
    del tmp
    gc.collect()

    # NUMBER OF TYPES OF PAST LOANS PER CUSTOMER
    tmp = data[["SK_ID_CURR", "CREDIT_TYPE"]].copy()
    tmp = tmp.groupby(["SK_ID_CURR"]).nunique()
    del tmp["SK_ID_CURR"]
    gc.collect()
    tmp.columns = ["CREDIT_TYPE_TYPES"]
    result = result.join(tmp, how="left")

    # AVERAGE NUMBER OF PAST LOANS PER TYPE PER CUSTOMER
    result["AVERAGE_LOAN_TYPE"] = result["HISTORY_CREDIT_COUNT"] * 1.0 / result["CREDIT_TYPE_TYPES"]

    # AVERAGE NUMBER OF DAYS BETWEEN SUCCESSIVE PAST APPLICATIONS FOR EACH CUSTOMER
    # Groupby each Customer and Sort values of DAYS_CREDIT in ascending order
    grp = data[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].copy().groupby(by=['SK_ID_CURR'])
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(drop=True)
    grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT'] * -1
    grp1['DAYS_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
    grp1['DAYS_DIFF'] = grp1['DAYS_DIFF'].fillna(0).astype('uint32')
    del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT']
    gc.collect()
    tmp = grp1[["SK_ID_CURR", "DAYS_DIFF"]].copy()
    tmp = helper(tmp.dropna(), "DAYS_DIFF")
    result = result.join(tmp, how="left")

    # % of LOANS PER CUSTOMER WHERE END DATE FOR CREDIT IS PAST
    tmp = data[["SK_ID_CURR", "DAYS_CREDIT_ENDDATE"]].copy()
    tmp["DAYS_CREDIT_ENDDATE"] = tmp["DAYS_CREDIT_ENDDATE"].apply(lambda x: 0 if (x < 0) else 1)
    tmp = tmp.groupby(["SK_ID_CURR"]).mean()
    tmp.columns = ["DAYS_CREDIT_ENDDATE_NEG_RATIO"]
    result = result.join(tmp, how="left")

    # AVERAGE NUMBER OF DAYS IN WHICH CREDIT EXPIRES IN FUTURE -INDICATION OF CUSTOMER DELINQUENCY IN FUTURE??
    data_tmp = data[data["DAYS_CREDIT_ENDDATE"] >= 0][["SK_ID_CURR", "DAYS_CREDIT_ENDDATE", "SK_ID_BUREAU"]].copy()
    # Groupby Each Customer ID
    grp = data_tmp.groupby(by=['SK_ID_CURR'])
    # Sort the values of CREDIT_ENDDATE for each customer ID
    grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE'], ascending=True)).reset_index(drop=True)
    del grp
    gc.collect()
    # Calculate the Difference in ENDDATES and fill missing values with zero
    grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE'].diff()
    grp1['DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
    data_tmp = data_tmp.merge(grp1[["SK_ID_BUREAU", "DAYS_ENDDATE_DIFF"]], on=["SK_ID_BUREAU"], how="left")
    del grp1
    gc.collect()
    tmp = data_tmp[["SK_ID_CURR", "DAYS_ENDDATE_DIFF"]].copy()
    tmp = helper(tmp.dropna(), "DAYS_ENDDATE_DIFF")
    result = result.join(tmp, how="left")

    # DEBT OVER CREDIT RATIO: The Ratio of Total Debt to Total Credit for each Customer
    result["DEBT_CREDIT_RATIO"] = result["AMT_CREDIT_SUM_DEBT_SUM"] * 1.0 / result["AMT_CREDIT_SUM_SUM"]

    # OVERDUE OVER DEBT RATIO
    result["OVERDUE_DEBT_RATIO"] = result["AMT_CREDIT_SUM_OVERDUE_SUM"] * 1.0 / result["AMT_CREDIT_SUM_DEBT_SUM"]

    # MONTHS_BALANCE 的统计特征
    tmp = data[["MONTHS_BALANCE", "SK_ID_CURR"]].copy()
    tmp = helper(tmp.dropna(), "MONTHS_BALANCE")
    result = result.join(tmp, how="left")

    # STATUS 是离散值，所以要做count，每类占比等特征
    tmp = data[["SK_ID_CURR", "STATUS"]].copy()
    tmp["STATUS"] = tmp["STATUS"].apply(lambda x: str(x))
    tmp = pd.get_dummies(tmp)
    tmp_count = tmp.groupby(["SK_ID_CURR"]).count()
    tmp_count.columns = [column + "_COUNT" for column in tmp_count.columns]
    tmp_mean = tmp.groupby(["SK_ID_CURR"]).mean()
    tmp_mean.columns = [column + "_AVG" for column in tmp_mean.columns]
    result = result.join(tmp_count, how="left").join(tmp_mean, how="left")

    if is_output:
        result.to_csv("home-credit-default-risk-preprocess/" + file_name)

    return result


def preprocess_pos_cash_balance(data, is_output=False, file_name=None):
    continue_names = ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]
    result = None
    for name in continue_names:
        tmp = data[["SK_ID_CURR", name]].copy()
        tmp = helper_stat(tmp.dropna(), name)
        if result is None:
            result = tmp
        else:
            result = result.join(tmp, how="left")

    discrete_names = ["NAME_CONTRACT_STATUS"]
    for name in discrete_names:
        tmp = data[["SK_ID_CURR", name]].copy().dropna()
        tmp = pd.get_dummies(tmp).groupby("SK_ID_CURR").mean()
        result = result.join(tmp, how="left")

    if is_output:
        result.to_csv("home-credit-default-risk-preprocess/" + file_name)

    return result


def preprocess_credit_card_balance(data, is_output=False, file_name=None):
    continue_names = ["MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT",
            "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT",
            "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE",
            "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT",
            "CNT_INSTALMENT_MATURE_CUM", "SK_DPD", "SK_DPD_DEF"]
    result = None
    for name in continue_names:
        tmp = data[["SK_ID_CURR", name]].copy()
        tmp = helper_stat(tmp.dropna(), name)
        if result is None:
            result = tmp
        else:
            result = result.join(tmp, how="left")

    discrete_names = ["NAME_CONTRACT_STATUS"]
    for name in discrete_names:
        tmp = data[["SK_ID_CURR", name]].copy().dropna()
        tmp = pd.get_dummies(tmp).groupby("SK_ID_CURR").mean()
        result = result.join(tmp, how="left")

    if is_output:
        result.to_csv("home-credit-default-risk-preprocess/" + file_name)

    return result


def preprocess_preious_application(data, is_output=False, file_name=None):
    continue_names = ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
                      "HOUR_APPR_PROCESS_START", "RATE_DOWN_PAYMENT", "RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED",
                      "DAYS_DECISION", "SELLERPLACE_AREA", "CNT_PAYMENT", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
                      "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    result = None
    for name in continue_names:
        tmp = data[["SK_ID_CURR", name]].copy()
        tmp = helper_stat(tmp.dropna(), name)
        if result is None:
            result = tmp
        else:
            result = result.join(tmp, how="left")

    discrete_names = ["NAME_CONTRACT_TYPE", "WEEKDAY_APPR_PROCESS_START", "FLAG_LAST_APPL_PER_CONTRACT", "NAME_CASH_LOAN_PURPOSE",
                      "NAME_CONTRACT_STATUS", "NAME_PAYMENT_TYPE", "CODE_REJECT_REASON", "NAME_TYPE_SUITE", "NAME_CLIENT_TYPE",
                      "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO", "NAME_PRODUCT_TYPE", "CHANNEL_TYPE", "NAME_SELLER_INDUSTRY",
                      "NAME_YIELD_GROUP", "PRODUCT_COMBINATION"]
    for name in discrete_names:
        tmp = data[["SK_ID_CURR", name]].copy().dropna()
        tmp = pd.get_dummies(tmp).groupby("SK_ID_CURR").mean()
        result = result.join(tmp, how="left")

    '''---------------------unused-------------------------'''
    # AMT_GOOD_PRICE 和 AMT_CREDIT 的比较
    data["GOOD_DIVIDE_CREDIT"] = data["AMT_GOODS_PRICE"] * 1.0 / data["AMT_CREDIT"]
    data["GOOD_MINUS_CREDIT"] = data["AMT_GOODS_PRICE"] - data["AMT_CREDIT"]
    for name in ["GOOD_DIVIDE_CREDIT", "GOOD_MINUS_CREDIT"]:
        tmp = data[["SK_ID_CURR", name]].copy()
        tmp = helper_stat(tmp.dropna(), name)
        result = result.join(tmp, how="left")

    # 计算用户历史application的待还款金额
    tmp = data[data["DAYS_TERMINATION"] < 0][["SK_ID_CURR", "DAYS_TERMINATION", "AMT_ANNUITY"]].copy()
    tmp = tmp.dropna()
    tmp["DEBT_AMOUNT"] = tmp["DAYS_TERMINATION"].apply(lambda x: int(x / 30.0 * -1) + 1) * tmp["AMT_ANNUITY"]
    tmp = helper_stat(tmp[["SK_ID_CURR", "DEBT_AMOUNT"]].dropna(), "DEBT_AMOUNT")
    result = result.join(tmp, how="left")

    # 统计状态为 Approved 的 ANNUITY
    tmp = data[data["NAME_CONTRACT_STATUS"] == "Approved"][["SK_ID_CURR", "AMT_ANNUITY"]].copy()
    tmp.columns = ["SK_ID_CURR", "AMT_ANNUITY_Approved"]
    tmp = helper_stat(tmp.dropna(), "AMT_ANNUITY_Approved")
    result = result.join(tmp, how="left")

    # 统计 DAYS_TERMINATION<0 部分的AMT_ANNUITY
    tmp = data[data["DAYS_TERMINATION"] < 0][["SK_ID_CURR", "AMT_ANNUITY"]].copy()
    tmp.columns = ["SK_ID_CURR", "AMT_ANNUITY_DEBT"]
    tmp = helper_stat(tmp.dropna(), "AMT_ANNUITY_DEBT")
    result = result.join(tmp, how="left")

    # 统计状态为 Approved 的 AMT_CREDIT
    tmp = data[data["NAME_CONTRACT_STATUS"] == "Approved"][["SK_ID_CURR", "AMT_CREDIT"]].copy()
    tmp.columns = ["SK_ID_CURR", "AMT_CREDIT_Approved"]
    tmp = helper_stat(tmp.dropna(), "AMT_CREDIT_Approved")
    result = result.join(tmp, how="left")

    # 统计 DAYS_TERMINATION<0 的 app 个数和占比
    tmp = data[["SK_ID_CURR", "DAYS_TERMINATION"]].copy()
    tmp["DEBT_APP"] = tmp["DAYS_TERMINATION"].apply(lambda x: 1 if x < 0 else 0)
    tmp_count = tmp[["SK_ID_CURR", "DEBT_APP"]].copy()
    tmp_count = tmp_count.groupby(["SK_ID_CURR"]).count()
    tmp_count.columns = ["DEBT_APP_COUNT"]
    tmp_ratio = tmp[["SK_ID_CURR", "DEBT_APP"]].copy()
    tmp_ratio = tmp_ratio.groupby(["SK_ID_CURR"]).mean()
    tmp_ratio.columns = ["DEBT_APP_RATIO"]
    result = result.join(tmp_count, how="left").join(tmp_ratio, how="left")

    # 统计 DAYS_TERMINATION<0 部分的 AMT_CREDIT
    tmp = data[data["DAYS_TERMINATION"] < 0][["SK_ID_CURR", "AMT_CREDIT"]].copy()
    tmp.columns = ["SK_ID_CURR", "AMT_CREDIT_DEBT"]
    tmp = helper_stat(tmp.dropna(), "AMT_CREDIT_DEBT")
    result = result.join(tmp, how="left")

    # 统计状态为 Approved 的 AMT_GOOD_PRICE
    tmp = data[data["NAME_CONTRACT_STATUS"] == "Approved"][["SK_ID_CURR", "AMT_GOODS_PRICE"]].copy()
    tmp.columns = ["SK_ID_CURR", "AMT_GOODS_PRICE_Approved"]
    tmp = helper_stat(tmp.dropna(), "AMT_GOODS_PRICE_Approved")
    result = result.join(tmp, how="left")

    # 统计 DAYS_TERMINATION<0 部分的 AMT_GOOD_PRICE
    tmp = data[data["DAYS_TERMINATION"] < 0][["SK_ID_CURR", "AMT_GOODS_PRICE"]].copy()
    tmp.columns = ["SK_ID_CURR", "AMT_GOODS_PRICE_DEBT"]
    tmp = helper_stat(tmp.dropna(), "AMT_GOODS_PRICE_DEBT")
    result = result.join(tmp, how="left")


    if is_output:
        result.to_csv("home-credit-default-risk-preprocess/" + file_name)
    return result


def preprocess_installments_payments(data, is_output=False, file_name=None):
    continue_names = ["NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                      "AMT_INSTALMENT", "AMT_PAYMENT"]
    result = None
    for name in continue_names:
        tmp = data[["SK_ID_CURR", name]].copy()
        tmp = helper_stat(tmp.dropna(), name)
        if result is None:
            result = tmp
        else:
            result = result.join(tmp, how="left")

    if is_output:
        result.to_csv("home-credit-default-risk-preprocess/" + file_name)
    return result


# 在join训练集之后做的填空值
def process_all_data(data):
    # SK_ID_CURR_Closed, SK_ID_CURR_Active, SK_ID_CURR_Sold, SK_ID_CURR_Bad debt
    name_lst = ["CREDIT_ACTIVE_Closed", "CREDIT_ACTIVE_Active", "CREDIT_ACTIVE_Sold", "CREDIT_ACTIVE_Bad debt",
                "CREDIT_DAY_OVERDUE_SUM", "CREDIT_DAY_OVERDUE_MAX", "CREDIT_DAY_OVERDUE_MEAN", "AMT_CREDIT_MAX_OVERDUE_SUM",
                "AMT_CREDIT_MAX_OVERDUE_MAX", "AMT_CREDIT_MAX_OVERDUE_MEAN"]
    for name in name_lst:
        data[name] = data[name].fillna(0)
    return data


def application_train_prewrapper(read_from_file=False, save_file=False, file_name=None):
    if read_from_file:
        return pd.read_csv("home-credit-default-risk-preprocess/" + file_name)
    else:
        application_train = pd.read_csv(data_dir + "application_train.csv")
        return proprocess_application_train_test(application_train, save_file, file_name)


def application_test_prewrapper(read_from_file=False, save_file=False, file_name=None):
    if read_from_file:
        return pd.read_csv("home-credit-default-risk-preprocess/" + file_name)
    else:
        application_test = pd.read_csv(data_dir + "application_test.csv")
        return proprocess_application_train_test(application_test, save_file, file_name)


def bureau_prewrapper(read_from_file=False, save_file=False, file_name=None):
    if read_from_file:
        return pd.read_csv("home-credit-default-risk-preprocess/" + file_name, index_col="SK_ID_CURR")
    else:
        bureau = pd.read_csv(data_dir + "bureau.csv")
        bureau_balance = pd.read_csv(data_dir + "bureau_balance.csv")
        data = pd.merge(bureau, bureau_balance, on=["SK_ID_BUREAU"], how="left")
        return preprocess_burau(data, save_file, file_name)


def pos_cash_balance_prewrapper(read_from_file=False, save_file=False, file_name=None):
    if read_from_file:
        return pd.read_csv("home-credit-default-risk-preprocess/" + file_name, index_col="SK_ID_CURR")
    else:
        pos_cash_balance = pd.read_csv(data_dir + "POS_CASH_balance.csv")
        return preprocess_pos_cash_balance(pos_cash_balance, save_file, file_name)


def credit_card_balance_prewrapper(read_from_file=False, save_file=False, file_name=None):
    if read_from_file:
        return pd.read_csv("home-credit-default-risk-preprocess/" + file_name, index_col="SK_ID_CURR")
    else:
        credit_card_balance = pd.read_csv(data_dir + "credit_card_balance.csv")
        return preprocess_credit_card_balance(credit_card_balance, save_file, file_name)


def previous_application_prewrapper(read_from_file=False, save_file=False, file_name=None):
    if read_from_file:
        return pd.read_csv("home-credit-default-risk-preprocess/" + file_name, index_col="SK_ID_CURR")
    else:
        previous_application = pd.read_csv(data_dir + "previous_application.csv")
        return preprocess_preious_application(previous_application, save_file, file_name)


def installments_payments_prewrapper(read_from_file=False, save_file=False, file_name=None):
    if read_from_file:
        return pd.read_csv("home-credit-default-risk-preprocess/" + file_name, index_col="SK_ID_CURR")
    else:
        installments_payments = pd.read_csv(data_dir + "installments_payments.csv")
        return preprocess_installments_payments(installments_payments, save_file, file_name)


'''------------------------------------------------构造特征-----------------------------------------------------------'''


# 基于训练集的一些与时间无关的统计信息，这些统计信息可以同时放到训练集和测试集中，作为预测不还款概率的先验
# 传入的是all_data，就是training数据和test数据整合以后的数据
def generate_all_data_features(data):
    # 计算单一字段下的还款率
    name_lst = ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
                "ORGANIZATION_TYPE"]
    for name in name_lst:
        all_sample = data[[name, "TARGET"]].dropna().groupby([name]).count()
        unable_sample = data[data["TARGET"] == 1][[name, "TARGET"]].dropna().groupby([name]).count()
        all_sample = all_sample.join(unable_sample, how="left", lsuffix="_all", rsuffix="_1")
        all_sample[name + "_TARGET_RATIO"] = all_sample["TARGET_1"] * 1.0 / all_sample["TARGET_all"]
        data = data.join(all_sample[[name + "_TARGET_RATIO"]], on=[name], how="left")
        data[name + "_TARGET_RATIO"] = data[name + "_TARGET_RATIO"].fillna(data[name + "_TARGET_RATIO"].mean())

    # 计算两个字段下的还款率
    name_lst = [["NAME_INCOME_TYPE", "NAME_FAMILY_STATUS"], ["NAME_INCOME_TYPE", "CNT_CHILDREN"],
                ["OCCUPATION_TYPE", "NAME_FAMILY_STATUS"], ["OCCUPATION_TYPE", "CNT_CHILDREN"],
                ["NAME_INCOME_TYPE", "CODE_GENDER"], ["OCCUPATION_TYPE", "CODE_GENDER"],
                ["REGION_RATING_CLIENT", "NAME_EDUCATION_TYPE"], ["REGION_RATING_CLIENT_W_CITY", "NAME_EDUCATION_TYPE"],
                ["REGION_RATING_CLIENT", "OCCUPATION_TYPE"], ["REGION_RATING_CLIENT_W_CITY", "OCCUPATION_TYPE"]]
    feature_names = ["INCOME_FAMILY_TARGET_RATIO", "INCOME_CHILDREN_TARGET_RATIO", "OCCUPATION_FAMILY_TARGET_RATIO",
                     "OCCUPATION_CHILDREN_TARGET_RATIO", "INCOME_GENDER_TARGET_RATIO", "OCCUPATION_GENDER_TARGET_RATIO",
                     "REGION_EDUCATION_TARGET_RATIO", "REGION_CITY_EDUCATION_TARGET_RATIO",
                     "REGION_OCCUPATION_TARGET_RATIO", "REGION_CITY_OCCUPATION_TARGET_RATIO"]
    for idx in range(len(name_lst)):
        names = name_lst[idx]
        feature_name = feature_names[idx]
        names.append("TARGET")
        all_sample = data[names].dropna().groupby(names[:-1]).count()
        unable_sample = data[data["TARGET"] == 1][names].dropna().groupby(names[:-1]).count()
        all_sample = all_sample.join(unable_sample, how="left", lsuffix="_all", rsuffix="_1")
        all_sample[feature_name] = all_sample["TARGET_1"] * 1.0 / all_sample["TARGET_all"]
        data = data.join(all_sample[feature_name], on=names[:-1], how="left")
        data[feature_name] = data[feature_name].fillna(data[feature_name].mean())

    # 增加用户贷款金额，收入，年金，商品价格在该贷款种类中所占的分为数情况
    def helper(quantile, value):
        for i in range(len(quantile)):
            if (i == 0) and (value < quantile[i]):
                return 0
            elif (i == len(quantile) - 1) and (value >= quantile[i]):
                return len(quantile)
            elif (value >= quantile[i]) and (value < quantile[i + 1]):
                return i

    cash_data = data[data["NAME_CONTRACT_TYPE"] == "Cash loans"].copy()
    rev_data = data[data["NAME_CONTRACT_TYPE"] == "Revolving loans"].copy()
    used_names = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    feature_names = ["CONTRACT_CREDIT_RANK", "CONTRACT_INCOME_RANK", "CONTRACT_ANNUITY_RANK", "CONTRACT_GOODS_RANK"]
    for idx in range(len(used_names)):
        used_name = used_names[idx]
        feature_name = feature_names[idx]
        cash_quantile = [cash_data[used_name].quantile((x + 1) * 0.01) for x in range(100)]
        rev_quantile = [rev_data[used_name].quantile((x + 1) * 0.01) for x in range(100)]
        cash_data[feature_name] = cash_data[used_name].apply(lambda x: helper(cash_quantile, x))
        rev_data[feature_name] = rev_data[used_name].apply(lambda x: helper(rev_quantile, x))
    data = cash_data.append(rev_data)

    # AMT_INCOME_TOTAL 和 AMT_ANNUITY 的交叉分位数特征，取其中一个特征按照数值等间隔划分，聚合后求另一个特征在聚合数据块内所在的分位数情况
    income_splits = [0, 5e4, 10e4, 15e4, 20e4, 25e4, 30e4, 35e4, 40e4, 10e11]
    income_quantile_data = [data[(data["AMT_INCOME_TOTAL"] >= income_splits[i]) &
                                 (data["AMT_INCOME_TOTAL"] < income_splits[i + 1])].copy()
                            for i in range(0, len(income_splits) - 1, 1)]
    tmp = pd.DataFrame()
    for quantile in income_quantile_data:
        annuity_quantile = [quantile["AMT_ANNUITY"].quantile(0.05 * i) for i in range(1, 21, 1)]
        quantile["INCOME_ANNUITY_RANK"] = quantile["AMT_ANNUITY"].apply(lambda x: helper(annuity_quantile, x))
        tmp = tmp.append(quantile)
    data = tmp
    annuity_splits = [0, 1.25e4, 2.5e4, 3.75e4, 5e4, 6.25e4, 7.5e4, 7.5e4, 10e11]
    annuity_quantile_data = [data[(data["AMT_ANNUITY"] >= annuity_splits[i]) &
                                  (data["AMT_ANNUITY"] < annuity_splits[i + 1])].copy()
                             for i in range(0, len(annuity_splits) - 1, 1)]
    tmp = pd.DataFrame()
    for quantile in annuity_quantile_data:
        income_quantile = [quantile["AMT_INCOME_TOTAL"].quantile(0.05 * i) for i in range(1, 21, 1)]
        quantile["ANNUITY_INCOME_RANK"] = quantile["AMT_INCOME_TOTAL"].apply(lambda x: helper(income_quantile, x))
        tmp = tmp.append(quantile)
    data = tmp

    # 计算用户当前贷款金额与数次历史贷款金额的对比情况
    data["CREDIT_LARGER_HISTORY_MEAN"] = (data["AMT_CREDIT"] - data["AMT_CREDIT_SUM_MEAN"]).apply(
        lambda x: 1 if (x > 0) else 0)
    data["CREDIT_LARGER_HISTORY_STD"] = (data["AMT_CREDIT"] - data["AMT_CREDIT_SUM_MEAN"] -
                                         data["AMT_CREDIT_SUM_STD"]).apply(lambda x: 1 if (x > 0) else 0)
    data["CREDIT_LARGER_HISTORY_2STD"] = (data["AMT_CREDIT"] - data["AMT_CREDIT_SUM_MEAN"] -
                                          2 * data["AMT_CREDIT_SUM_STD"]).apply(lambda x: 1 if (x > 0) else 0)
    data["CREDIT_SMALLER_HISTORY_STD"] = (data["AMT_CREDIT"] - data["AMT_CREDIT_SUM_MEAN"] +
                                          data["AMT_CREDIT_SUM_STD"]).apply(lambda x: 1 if (x > 0) else 0)
    data["CREDIT_SMALLER_HISTORY_2STD"] = (data["AMT_CREDIT"] - data["AMT_CREDIT_SUM_MEAN"] +
                                        2 * data["AMT_CREDIT_SUM_STD"]).apply(lambda x: 1 if (x > 0) else 0)

    # AMT_CREDIT_SUM_DEBT_SUM 和当前application贷款金额做对比（差值或者比值）
    data["AMT_CREDIT_SUM_DEBT_SUM"] = data['AMT_CREDIT_SUM_DEBT_SUM'].fillna(0)
    data["HISTORY_DEBT_MINUS_CREDIT"] = data["AMT_CREDIT_SUM_DEBT_SUM"] - data["AMT_CREDIT"]
    data["HISTORY_DEBT_DIVIDE_CREDIT"] = data["AMT_CREDIT_SUM_DEBT_SUM"] * 1.0 / data["AMT_CREDIT"]
    data["HISTORY_DEBT_ADD_CREDIT"] = data["AMT_CREDIT_SUM_DEBT_SUM"] + data["AMT_CREDIT"]
    data["HISTORY_LOG_DEBT_ADD_CREDIT"] = np.log1p(data["AMT_CREDIT_SUM_DEBT_SUM"]) + data["AMT_CREDIT"]
    data["HISTORY_DEBT_ADD_LOG_CREDIT"] = data["AMT_CREDIT_SUM_DEBT_SUM"] + np.log1p(data["AMT_CREDIT"])


    return data


def feature_after_dummy(data):
    # 用application_train里除了ex_source_x以外的所有数据做特征，用ex_source_x做标签，填充空值
    application_train_feature = read_feature("application_train_feature")
    ori_names = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    new_names = ["EXT_SOURCE_1_FILL_XGB", "EXT_SOURCE_2_FILL_XGB", "EXT_SOURCE_1_FILL_XGB"]
    for idx in range(len(ori_names)):
        name = ori_names[idx]
        new_name = new_names[idx]
        train_X = data[~data[name].isnull()][application_train_feature].drop([name], axis=1)
        train_y = data[~data[name].isnull()][name]
        quantile = train_y.quantile(0.2)
        train_y = train_y.apply(lambda x: 0 if x <= quantile else 1)
        test_X = data[data[name].isnull()][application_train_feature].drop([name], axis=1)
        train_ds = lgb.Dataset(train_X, label=train_y)
        lgb_model = lgb.train(lgb_param_reg, train_ds, num_boost_round=1000, verbose_eval=50)
        test_y = lgb_model.predict(test_X)
        unnull_part = data[~data[name].isnull()].copy()
        unnull_part[new_name] = unnull_part[name]
        null_part = data[data[name].isnull()].copy()
        null_part[new_name] = test_y
        data = unnull_part.append(null_part)
    return data


def generate_features(data):
    # 20180602：
    # 增加特征：AMT_ANNUITY和AMT_INCOME_TOTAL的匹配情况
    data["INCOME_MINUS_ANNUITY"] = data["AMT_INCOME_TOTAL"] - data["AMT_ANNUITY"]
    data["INCOME_DIVIDE_ANNUITY"] = data["AMT_INCOME_TOTAL"] * 1.0 / data["AMT_ANNUITY"]

    # 增加特征：AMT_CREDIT和AMT_INCOME_TOTAL的匹配情况
    data["CREDIT_MINUS_INCOME"] = data["AMT_CREDIT"] - data["AMT_INCOME_TOTAL"]
    data["CREDIT_DIVIDE_INCOME"] = data["AMT_CREDIT"] * 1.0 / data["AMT_INCOME_TOTAL"]

    # 增加特征：AMT_CREDIT和AMT_GOODS_PRICE的匹配情况
    data["GOOD_MINUS_CREDIT"] = data["AMT_GOODS_PRICE"] - data["AMT_CREDIT"]
    data["GOOD_DIVIDE_CREDIT"] = data["AMT_GOODS_PRICE"] * 1.0 / data["AMT_CREDIT"]

    # 增加特征：贷款的期数
    data["CREDIT_DIVIDE_ANNUITY"] = data["AMT_CREDIT"] * 1.0 / data["AMT_ANNUITY"]

    # 20180603
    # 构造用户提交文件的个数
    name_lst = ["FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6",
                "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11",
                "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15" , "FLAG_DOCUMENT_16",
                "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"]
    data["DOCUMENT_COUNT"] = data["FLAG_DOCUMENT_2"]
    for name in name_lst:
        data["DOCUMENT_COUNT"] += data[name]

    # 针对用户永久地址是否匹配，计算不匹配次数
    data["ADDRESS_NOT_MATCH_COUNT"] = data["REG_REGION_NOT_LIVE_REGION"] + data["REG_REGION_NOT_WORK_REGION"] + data["LIVE_REGION_NOT_WORK_REGION"]
    data["CITY_NOT_MATCH_COUNT"] = data["REG_CITY_NOT_LIVE_CITY"] + data["REG_CITY_NOT_WORK_CITY"] + data["LIVE_CITY_NOT_WORK_CITY"]

    # 各 OCCUPATION_TYPE 和 REGION_RATING_CLIENT_W_CITY（或者REGION_RATING_CLIENT）里 AMT_INCOME_TOTAL，
    # AMT_CREDIT，AMT_ANNUITY，AMT_GOODS_PRICE 的均值，中位数
    # 再计算 AMT_INCOME_TOTAL，AMT_CREDIT，AMT_ANNUITY，AMT_GOODS_PRICE 和对应的均值、中位数的差值，比值
    # 均值
    tmp = data[["OCCUPATION_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                "REGION_RATING_CLIENT_W_CITY"]].groupby(
        ["OCCUPATION_TYPE", "REGION_RATING_CLIENT_W_CITY"]).mean()
    tmp.columns = ["INCOME_OCCUPATION_CITY_MEAN", "CREDIT_OCCUPATION_CITY_MEAN", "ANNUITY_OCCUPATION_CITY_MEAN",
                   "GOOD_PRICE_OCCUPATION_CITY_MEAN"]
    data = data.join(tmp, on=["OCCUPATION_TYPE", "REGION_RATING_CLIENT_W_CITY"], how="left")
    data["INCOME_MINUS_OCCUPATION_CITY_MEAN"] = data["AMT_INCOME_TOTAL"] - data["INCOME_OCCUPATION_CITY_MEAN"]
    data["CREDIT_MINUS_OCCUPATION_CITY_MEAN"] = data["AMT_CREDIT"] - data["CREDIT_OCCUPATION_CITY_MEAN"]
    data["ANNUITY_MINUS_OCCUPATION_CITY_MEAN"] = data["AMT_ANNUITY"] - data["ANNUITY_OCCUPATION_CITY_MEAN"]
    data["GOOD_PRICE_MINUS_OCCUPATION_CITY_MEAN"] = data["AMT_GOODS_PRICE"] - data["GOOD_PRICE_OCCUPATION_CITY_MEAN"]
    data["INCOME_DIVIDE_OCCUPATION_CITY_MEAN"] = data["AMT_INCOME_TOTAL"] * 1.0 / data["INCOME_OCCUPATION_CITY_MEAN"]
    data["CREDIT_DIVIDE_OCCUPATION_CITY_MEAN"] = data["AMT_CREDIT"] * 1.0 / data["CREDIT_OCCUPATION_CITY_MEAN"]
    data["ANNUITY_DIVIDE_OCCUPATION_CITY_MEAN"] = data["AMT_ANNUITY"] * 1.0 / data["ANNUITY_OCCUPATION_CITY_MEAN"]
    data["GOOD_PRICE_DIVIDE_OCCUPATION_CITY_MEAN"] = data["AMT_GOODS_PRICE"] * 1.0 / data["GOOD_PRICE_OCCUPATION_CITY_MEAN"]

    tmp = data[["OCCUPATION_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                "REGION_RATING_CLIENT"]].groupby(
        ["OCCUPATION_TYPE", "REGION_RATING_CLIENT"]).mean()
    tmp.columns = ["INCOME_OCCUPATION_REG_MEAN", "CREDIT_OCCUPATION_REG_MEAN", "ANNUITY_OCCUPATION_REG_MEAN",
                   "GOOD_PRICE_OCCUPATION_REG_MEAN"]
    data = data.join(tmp, on=["OCCUPATION_TYPE", "REGION_RATING_CLIENT"], how="left")
    data["INCOME_MINUS_OCCUPATION_REG_MEAN"] = data["AMT_INCOME_TOTAL"] - data["INCOME_OCCUPATION_REG_MEAN"]
    data["CREDIT_MINUS_OCCUPATION_REG_MEAN"] = data["AMT_CREDIT"] - data["CREDIT_OCCUPATION_REG_MEAN"]
    data["ANNUITY_MINUS_OCCUPATION_REG_MEAN"] = data["AMT_ANNUITY"] - data["ANNUITY_OCCUPATION_REG_MEAN"]
    data["GOOD_PRICE_MINUS_OCCUPATION_REG_MEAN"] = data["AMT_GOODS_PRICE"] - data["GOOD_PRICE_OCCUPATION_REG_MEAN"]
    data["INCOME_DIVIDE_OCCUPATION_REG_MEAN"] = data["AMT_INCOME_TOTAL"] * 1.0 / data["INCOME_OCCUPATION_REG_MEAN"]
    data["CREDIT_DIVIDE_OCCUPATION_REG_MEAN"] = data["AMT_CREDIT"] * 1.0 / data["CREDIT_OCCUPATION_REG_MEAN"]
    data["ANNUITY_DIVIDE_OCCUPATION_REG_MEAN"] = data["AMT_ANNUITY"] * 1.0 / data["ANNUITY_OCCUPATION_REG_MEAN"]
    data["GOOD_PRICE_DIVIDE_OCCUPATION_REG_MEAN"] = data["AMT_GOODS_PRICE"] * 1.0 / data[
        "GOOD_PRICE_OCCUPATION_REG_MEAN"]
    # data = data.drop(["INCOME_OCCUPATION_CITY_MEAN", "CREDIT_OCCUPATION_CITY_MEAN", "ANNUITY_OCCUPATION_CITY_MEAN",
    #                "GOOD_PRICE_OCCUPATION_CITY_MEAN"], axis=1)
    # 中位数
    tmp = data[["OCCUPATION_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                "REGION_RATING_CLIENT_W_CITY"]].groupby(["OCCUPATION_TYPE", "REGION_RATING_CLIENT_W_CITY"]).median()
    tmp.columns = ["INCOME_OCCUPATION_CITY_MEDI", "CREDIT_OCCUPATION_CITY_MEDI", "ANNUITY_OCCUPATION_CITY_MEDI",
                   "GOOD_PRICE_OCCUPATION_CITY_MEDI"]
    data = data.join(tmp, on=["OCCUPATION_TYPE", "REGION_RATING_CLIENT_W_CITY"], how="left")
    data["INCOME_MINUS_OCCUPATION_CITY_MEDI"] = data["AMT_INCOME_TOTAL"] - data["INCOME_OCCUPATION_CITY_MEDI"]
    data["CREDIT_MINUS_OCCUPATION_CITY_MEDI"] = data["AMT_CREDIT"] - data["CREDIT_OCCUPATION_CITY_MEDI"]
    data["ANNUITY_MINUS_OCCUPATION_CITY_MEDI"] = data["AMT_ANNUITY"] - data["ANNUITY_OCCUPATION_CITY_MEDI"]
    data["GOOD_PRICE_MINUS_OCCUPATION_CITY_MEDI"] = data["AMT_GOODS_PRICE"] - data["GOOD_PRICE_OCCUPATION_CITY_MEDI"]
    data["INCOME_DIVIDE_OCCUPATION_CITY_MEDI"] = data["AMT_INCOME_TOTAL"] * 1.0 / data["INCOME_OCCUPATION_CITY_MEDI"]
    data["CREDIT_DIVIDE_OCCUPATION_CITY_MEDI"] = data["AMT_CREDIT"] * 1.0 / data["CREDIT_OCCUPATION_CITY_MEDI"]
    data["ANNUITY_DIVIDE_OCCUPATION_CITY_MEDI"] = data["AMT_ANNUITY"] * 1.0 / data["ANNUITY_OCCUPATION_CITY_MEDI"]
    data["GOOD_PRICE_DIVIDE_OCCUPATION_CITY_MEDI"] = data["AMT_GOODS_PRICE"] * 1.0 / data["GOOD_PRICE_OCCUPATION_CITY_MEDI"]
    # data = data.drop(["INCOME_OCCUPATION_CITY_MEDI", "CREDIT_OCCUPATION_CITY_MEDI", "ANNUITY_OCCUPATION_CITY_MEDI",
    #                "GOOD_PRICE_OCCUPATION_CITY_MEDI"], axis=1)
    tmp = data[["OCCUPATION_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                "REGION_RATING_CLIENT"]].groupby(["OCCUPATION_TYPE", "REGION_RATING_CLIENT"]).median()
    tmp.columns = ["INCOME_OCCUPATION_REG_MEDI", "CREDIT_OCCUPATION_REG_MEDI", "ANNUITY_OCCUPATION_REG_MEDI",
                   "GOOD_PRICE_OCCUPATION_REG_MEDI"]
    data = data.join(tmp, on=["OCCUPATION_TYPE", "REGION_RATING_CLIENT"], how="left")
    data["INCOME_MINUS_OCCUPATION_REG_MEDI"] = data["AMT_INCOME_TOTAL"] - data["INCOME_OCCUPATION_REG_MEDI"]
    data["CREDIT_MINUS_OCCUPATION_REG_MEDI"] = data["AMT_CREDIT"] - data["CREDIT_OCCUPATION_REG_MEDI"]
    data["ANNUITY_MINUS_OCCUPATION_REG_MEDI"] = data["AMT_ANNUITY"] - data["ANNUITY_OCCUPATION_REG_MEDI"]
    data["GOOD_PRICE_MINUS_OCCUPATION_REG_MEDI"] = data["AMT_GOODS_PRICE"] - data["GOOD_PRICE_OCCUPATION_REG_MEDI"]
    data["INCOME_DIVIDE_OCCUPATION_REG_MEDI"] = data["AMT_INCOME_TOTAL"] * 1.0 / data["INCOME_OCCUPATION_REG_MEDI"]
    data["CREDIT_DIVIDE_OCCUPATION_REG_MEDI"] = data["AMT_CREDIT"] * 1.0 / data["CREDIT_OCCUPATION_REG_MEDI"]
    data["ANNUITY_DIVIDE_OCCUPATION_REG_MEDI"] = data["AMT_ANNUITY"] * 1.0 / data["ANNUITY_OCCUPATION_REG_MEDI"]
    data["GOOD_PRICE_DIVIDE_OCCUPATION_REG_MEDI"] = data["AMT_GOODS_PRICE"] * 1.0 / data[
        "GOOD_PRICE_OCCUPATION_REG_MEDI"]

    #  AMT_CREDIT > AMT_GOODS_PRICE 说明该用户购买了信用保险，如果不能按时还钱，HOME CREDIT可以找保险公司所要赔偿
    data["CREDIT_MINUS_GOOD_PRICE"] = data["AMT_CREDIT"] - data["AMT_GOODS_PRICE"]
    data["CREDIT_DIVIDE_GOOD_PRICE"] = data["AMT_CREDIT"] * 1.0 / data["AMT_GOODS_PRICE"]
    data["CREDIT_LARGER_GOOD_PRICE"] = (data["AMT_CREDIT"] - data["AMT_GOODS_PRICE"]).apply(lambda x: 1 if (x > 0) else 0)

    # source计算分为点，然后将source转换成对应的分位点，对三个分位点计算均值，方差，等统计特征
    def helper(item, quantile):
        if item == quantile[0]:
            return 0
        for i in range(len(quantile) - 1):
            if (item > quantile[i]) and (item <= quantile[i + 1]):
                return i
        return len(quantile)
    source1_quantile = data["EXT_SOURCE_1"].quantile([i * 0.01 + 0.01 for i in range(99)]).values
    source2_quantile = data["EXT_SOURCE_2"].quantile([i * 0.01 + 0.01 for i in range(99)]).values
    source3_quantile = data["EXT_SOURCE_3"].quantile([i * 0.01 + 0.01 for i in range(99)]).values
    data["new_EXT_SOURCE_1"] = data["EXT_SOURCE_1"].apply(lambda x: helper(x, source1_quantile))
    data["new_EXT_SOURCE_2"] = data["EXT_SOURCE_2"].apply(lambda x: helper(x, source2_quantile))
    data["new_EXT_SOURCE_3"] = data["EXT_SOURCE_3"].apply(lambda x: helper(x, source3_quantile))
    data["new_EXT_SOURCE_MAX"] = data[["new_EXT_SOURCE_1", "new_EXT_SOURCE_2", "new_EXT_SOURCE_3"]].max(axis=1)
    data["new_EXT_SOURCE_MIN"] = data[["new_EXT_SOURCE_1", "new_EXT_SOURCE_2", "new_EXT_SOURCE_3"]].min(axis=1)
    data["new_EXT_SOURCE_MEAN"] = data[["new_EXT_SOURCE_1", "new_EXT_SOURCE_2", "new_EXT_SOURCE_3"]].mean(axis=1)
    data["new_EXT_SOURCE_STD"] = data[["new_EXT_SOURCE_1", "new_EXT_SOURCE_2", "new_EXT_SOURCE_3"]].std(axis=1)
    data["new_EXT_SOURCE_MEDIAN"] = data[["new_EXT_SOURCE_1", "new_EXT_SOURCE_2", "new_EXT_SOURCE_3"]].median(axis=1)
    data["new_EXT_SOURCE_SUM"] = data[["new_EXT_SOURCE_1", "new_EXT_SOURCE_2", "new_EXT_SOURCE_3"]].sum(axis=1)

    '''---------------------unused-------------------------'''
    # 从 previous_application 里计算用户当前负债程度
    data['PREVIOUS_APP_DEBT_DIVIDE_CREDIT'] = data["DEBT_AMOUNT_SUM"] * 1.0 / data["AMT_CREDIT_SUM"]
    data['PREVIOUS_APP_DEBT_MINUS_CREDIT'] = data["DEBT_AMOUNT_SUM"] - data["AMT_CREDIT_SUM"]

    return data


'''----------------------------------------------交叉验证-------------------------------------------------------------'''


baseline_feature = read_feature("baseline_feature")
# experiment_features = ["GOOD_DIVIDE_CREDIT", "GOOD_MINUS_CREDIT", "DEBT_AMOUNT", "AMT_ANNUITY_Approved",
#                        "AMT_ANNUITY_DEBT", "AMT_CREDIT_Approved", "AMT_CREDIT_DEBT", "AMT_GOODS_PRICE_Approved",
#                        "AMT_GOODS_PRICE_DEBT"]
# stat_suffix = ["MAX", "MIN", "MEAN", "STD", "MEDIAN", "SUM"]
# experiment_features = [[feature_name + "_" + suffix for suffix in stat_suffix] for feature_name in experiment_features]
# experiment_features.append(["DEBT_APP_COUNT", "DEBT_APP_RATIO"])
# experiment_features.append(["PREVIOUS_APP_DEBT_DIVIDE_CREDIT", "PREVIOUS_APP_DEBT_MINUS_CREDIT"])
# experiment_features.append(["SK_DPD_DEF_MAX_credit_card", "SK_DPD_DEF_MIN_credit_card", "SK_DPD_DEF_MEAN_credit_card",
#                             "SK_DPD_DEF_STD_credit_card", "SK_DPD_DEF_MEDIAN_credit_card", "SK_DPD_DEF_SUM_credit_card"])
experiment_features = [['GOOD_MINUS_CREDIT_MAX', 'GOOD_MINUS_CREDIT_MIN', 'GOOD_MINUS_CREDIT_MEAN', 'GOOD_MINUS_CREDIT_STD', 'GOOD_MINUS_CREDIT_MEDIAN', 'GOOD_MINUS_CREDIT_SUM', 'PREVIOUS_APP_DEBT_DIVIDE_CREDIT', 'PREVIOUS_APP_DEBT_MINUS_CREDIT', 'AMT_GOODS_PRICE_DEBT_MAX', 'AMT_GOODS_PRICE_DEBT_MIN', 'AMT_GOODS_PRICE_DEBT_MEAN', 'AMT_GOODS_PRICE_DEBT_STD', 'AMT_GOODS_PRICE_DEBT_MEDIAN', 'AMT_GOODS_PRICE_DEBT_SUM']]
def cross_validation_lgb(train_data, categorical_feature=[]):
    result_lst = []
    train_target1 = train_data[train_data["TARGET"] == 1].copy()
    train_target0 = train_data[train_data["TARGET"] == 0].copy()
    X_target1 = train_target1.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y_target1 = train_target1.TARGET
    X_target0 = train_target0.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y_target0 = train_target0.TARGET
    X_target1, y_target1 = shuffle(X_target1, y_target1)
    X_target0, y_target0 = shuffle(X_target0, y_target0)
    for append_feature in experiment_features:
        baseline_results = []
        ex_results = []
        ex_features = copy.deepcopy(baseline_feature)
        ex_features.extend(append_feature)
        for i in range(10):
            X_target1_train, X_target1_test, y_target1_train, y_target1_test = train_test_split(X_target1, y_target1, test_size=0.2,
                                                                                               random_state=i)
            X_target0_train, X_target0_test, y_target0_train, y_target0_test = train_test_split(X_target0, y_target0, test_size=0.2,
                                                                                               random_state=i)
            X_train = X_target0_train.append(X_target1_train)
            y_train = y_target0_train.append(y_target1_train)
            X_validation = X_target0_test.append(X_target1_test)
            y_validation = y_target0_test.append(y_target1_test)
            baseline_X_train = X_train[baseline_feature]
            baseline_X_validation = X_validation[baseline_feature]
            ex_X_train = X_train[ex_features]
            ex_X_validation = X_validation[ex_features]
            # baseline
            train_base = lgb.Dataset(data=baseline_X_train, label=y_train, free_raw_data=True)
            valid_base = lgb.Dataset(data=baseline_X_validation, label=y_validation, free_raw_data=True)
            lgb_base_model = lgb.train(lgb_param, train_base, valid_sets=valid_base, verbose_eval=50,
                                     num_boost_round=num_iterations, categorical_feature=categorical_feature)
            baseline_results.append(lgb_base_model.best_score.get("valid_0").get("auc"))
            # experiment with new feature
            train_ex = lgb.Dataset(data=ex_X_train, label=y_train, free_raw_data=True)
            valid_ex = lgb.Dataset(data=ex_X_validation, label=y_validation, free_raw_data=True)
            lgb_ex_model = lgb.train(lgb_param, train_ex, valid_sets=valid_ex, verbose_eval=50,
                                     num_boost_round=num_iterations, categorical_feature=categorical_feature)
            ex_results.append(lgb_ex_model.best_score.get("valid_0").get("auc"))
            # experiment with new parameter
            # lgb_ex_model = lgb.train(lgb_param_ex, train_base, valid_sets=valid_base, verbose_eval=50,
            #                          num_boost_round=num_iterations, categorical_feature=categorical_feature)
            # ex_results.append(lgb_ex_model.best_score.get("valid_0").get("auc"))
        result_lst.append([append_feature, np.mean(baseline_results), np.mean(ex_results),
                           np.std(baseline_results), np.std(ex_results),
                           np.mean(np.array(ex_results) - np.array(baseline_results)),
                           np.std(np.array(ex_results) - np.array(baseline_results))])
    return result_lst


'''----------------------------------------------训练模型-------------------------------------------------------------'''
used_features = copy.deepcopy(baseline_feature)

drop_columns = []
# categorical_feature = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
#                        "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
#                        "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
#                        "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
#                        "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]
categorical_feature = []


def training_lgb(train_data, categorical_feature=[]):
    X_train = train_data[used_features]
    y_train = train_data.TARGET
    train_ds = lgb.Dataset(data=X_train, label=y_train, free_raw_data=True)
    lgb_model = lgb.train(lgb_param, train_ds, valid_sets=train_ds, verbose_eval=50,
                          num_boost_round=num_iterations, categorical_feature=categorical_feature)
    return lgb_model


def mapping_category(dataframe, categorical_feature):
    for name in categorical_feature:
        value = dataframe[name].unique()
        dataframe[name] = dataframe[name].apply(lambda item: np.argwhere(value == item)[0][0])
    return dataframe


'''----------------------------------------------- main -------------------------------------------------------------'''
def generate_training_data():
    def shape_helper(data):
        print "shape of training data is " + str(data.shape)
        null_count = data.isnull().sum()
        print "null samples count : "
        print null_count[null_count > 0]

    train_data = application_train_prewrapper(read_from_file=True, save_file=False,
                                              file_name="application_train_preprocess.csv")

    shape_helper(train_data)
    test_data = application_test_prewrapper(read_from_file=True, save_file=False,
                                            file_name="application_test_preprocess.csv")
    shape_helper(test_data)
    bureau_data = bureau_prewrapper(read_from_file=True, save_file=False, file_name="bureau_preprocess.csv")
    shape_helper(bureau_data)
    pos_cash_balance_data = pos_cash_balance_prewrapper(read_from_file=True, save_file=False,
                                                        file_name="pos_cash_balance_preprocess.csv")
    shape_helper(pos_cash_balance_data)
    credit_card_balance_data = credit_card_balance_prewrapper(read_from_file=True, save_file=False,
                                                              file_name="credit_card_balance_preprocess.csv")
    shape_helper(credit_card_balance_data)
    previous_application_data = previous_application_prewrapper(read_from_file=True, save_file=False,
                                                                file_name="previous_application_preprocess.csv")
    shape_helper(previous_application_data)
    installments_payments_data = installments_payments_prewrapper(read_from_file=True, save_file=False,
                                                                  file_name="installments_payments_preprocess.csv")
    shape_helper(installments_payments_data)

    all_data = train_data.append(test_data)

    # 各种join
    all_data = all_data.join(bureau_data, on=["SK_ID_CURR"], how="left")
    all_data = all_data.join(pos_cash_balance_data, on=["SK_ID_CURR"], how="left", rsuffix="_pos")
    all_data = all_data.join(credit_card_balance_data, on=["SK_ID_CURR"], how="left", rsuffix="_credit_card")
    all_data = all_data.join(previous_application_data, on=["SK_ID_CURR"], how="left", rsuffix="_previous")
    all_data = all_data.join(installments_payments_data, on=["SK_ID_CURR"], how="left", rsuffix="_install")

    # 构建统一的特征
    all_data = generate_features(all_data)
    # 构建特征以后填空值
    # all_data = process_all_data(all_data)
    all_data = generate_all_data_features(all_data)
    if len(drop_columns) > 0:
        all_data = all_data.drop(drop_columns, axis=1)
    if len(categorical_feature) > 0:
        all_data = mapping_category(all_data, categorical_feature)
    else:
        all_data = pd.get_dummies(all_data)
        # all_data = feature_after_dummy(all_data)
    train_data = all_data[~all_data["TARGET"].isnull()]
    print "shape of training data after dummy is " + str(train_data.shape)
    test_data = all_data[all_data["TARGET"].isnull()]
    print "shape of test data after dummy is " + str(test_data.shape)

    return train_data, test_data

train_data, test_data = generate_training_data()

validation_result = cross_validation_lgb(train_data, categorical_feature)
for lst in validation_result:
    feature_name = lst[0]
    baseline_mean = lst[1]
    experiment_mean = lst[2]
    baseline_std = lst[3]
    experiment_std = lst[4]
    minus_mean = lst[5]
    minus_std = lst[6]
    print "feature names are:" + str(feature_name)
    print "auc avg of baseline and experiemnt are : baseline - %f, experiment - %f" % (baseline_mean, experiment_mean)
    print "std avg of baseline and experiemnt are : baseline - %f, experiment - %f" % (baseline_std, experiment_std)
    print "avg of experiment auc minus baseline auc is : %f" % minus_mean
    print "std of experiment auc minus baseline auc is : %f" % minus_std

used_features.extend(experiment_features[0])
lgb_model = training_lgb(train_data, categorical_feature)
result_df = test_data[["SK_ID_CURR"]].copy()
result_df["TARGET"] = lgb_model.predict(test_data[used_features])
result_df.to_csv("./result/20180707/result_3.csv", index=None)



lgb.plot_importance(lgb_model, height=0.5, max_num_features=20, ignore_zero=False, figsize=(20,6),
                    importance_type ='gain')
plt.show()