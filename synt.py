"""This Script loads all the features extracted from the pairs of sentences, train the models, 
and predicts the spearman score on the test set provided by the organizers."""

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import numpy as np 
import pandas as pd
import spacy
from scipy import stats

# LOADING DATASETS 

raw = pd.read_csv("eng_train.csv")
raw2 = pd.read_csv("eng_dev_with_labels.csv")
raw3 = pd.read_csv("eng_test.csv")
X = raw["Text"]
X2 = raw2["Text"]
X3 = raw3["Text"]

# LOADING FEATURES 

bert_score = np.load("pap_res_vo_train.npz", allow_pickle=True)
bert_score = bert_score["arr_0"]
#bert_score2 = pd.read_csv("sbert_score_dev_labels.txt").to_numpy()
#bert_score3 = np.load("sbert_score_test.npz")
#bert_score = np.load("albert_score_train.npz")
bert_score2 = np.load("pap_res_vo_dev_lab.npz", allow_pickle=True)
#bert_score3 = np.load("albert_score_test.npz")
#bert_score = bert_score["arr_0"]
bert_score2 = bert_score2["arr_0"]
#print(np.isnan(bert_score2).any())
#bert_score3 = bert_score3["arr_0"]
scores = raw2["Score"]
#scores2 = raw2["Score"]
scores = scores
features = np.load("pap_synctact_train.npz", allow_pickle=True)
features2 = np.load("pap_synctact_dev_lab.npz", allow_pickle=True)
luis_train = np.loadtxt("luis_train_v2.txt")
luis_dev = np.loadtxt("luis_val_v2.txt")
luis_test = np.loadtxt("luis_test_v2.txt")

feat = features["arr_0"]
feat2 = features2["arr_0"]
leven = np.load("pap_leven_score_train.npz", allow_pickle=True)
leven2 = np.load("pap_leven_score_dev_lab.npz", allow_pickle=True)
lev = leven["arr_0"]
lev2 = leven2["arr_0"]

# Convert feat array to DataFrame with the new column
feat_df = pd.DataFrame(feat, columns=[f'feature_{i+1}' for i in range(feat.shape[1])])
feat_df2 = pd.DataFrame(feat2, columns=[f'feature_{i+1}' for i in range(feat.shape[1])])
feat_df3 = pd.DataFrame(luis_train, columns=[f'feature_{4+i}' for i in range(luis_train.shape[1])])
feat_df4 = pd.DataFrame(luis_dev, columns=[f'feature_{4+i}' for i in range(luis_dev.shape[1])])
#feat_df3 = pd.DataFrame(feat3, columns=[f'feature_{i+1}' for i in range(feat.shape[1])])
feat_df['bert_score'] = bert_score

feat_df2['bert_score'] = bert_score2
feat_df["leven"] = lev
feat_df2["leven"] = lev2

x = feat_df[["feature_1", "feature_2", "feature_3", "bert_score", "leven" ]].to_numpy()
merged_x_train = pd.concat([feat_df, feat_df3], axis=1)
merged_x_train = merged_x_train.to_numpy()

x2 = feat_df2[["feature_1", "feature_2", "feature_3", "bert_score", "leven" ]].to_numpy()
merged_dev = pd.concat([feat_df2, feat_df4 ], axis=1)
merged_dev = merged_dev.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(merged_dev, scores, test_size=0.2, random_state=42)
#X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, scores2, test_size=0.2, random_state=42)
model4 = LinearRegression()
model4.fit(X_train, y_train)
y_preds = model4.predict(X_test)
y_preds2 = model4.predict(merged_dev)
mse = mean_squared_error(y_test, y_preds)
res = stats.spearmanr(np.array(y_test), np.array(y_preds))
#res_dev = stats.spearmanr(np.array(scores2), np.array(y_preds2))

#-- MODEL TRAININGS AND PREDICTIONS

# Gradient Boost Reg 
g_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

g_model.fit(X_train, y_train)
y_gpreds = g_model.predict(X_test)
y_final_preds = g_model.predict(merged_dev)
pair_ids = [f"ENG-test-{str(i).zfill(4)}" for i in range(0000, 0000 + len(y_final_preds))]
#res = pd.DataFrame({"pairid": pair_ids ,"pred_score": y_final_preds})
#res.to_csv("submit.csv", index=False)
mse2 = mean_squared_error(y_test, y_gpreds)
res2 = stats.spearmanr(np.array(y_test), np.array(y_gpreds))
#res_dev2 = stats.spearmanr(np.array(scores2), np.array(y_final_preds))

# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
        'objective': "reg:squarederror",
        'colsample_bytree': 0.8,
        'learning_rate': 0.1,
        'max_depth': 3,
        'alpha':0.2,
        'n_estimators':100,
        'random_state': 42
        }

xgb_model = xgb.train(params, dtrain)
y_xpreds = xgb_model.predict(dtest)

mse3 = mean_squared_error(y_test, y_xpreds)
res3 = stats.spearmanr(np.array(y_test), np.array(y_xpreds))

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_fpreds = rf_model.predict(X_test)
y_dev_pred = rf_model.predict(merged_dev)
mse4 = mean_squared_error(y_test, y_fpreds)
res4 = stats.spearmanr(np.array(y_test), np.array(y_fpreds))
#res_dev3 = stats.spearmanr(np.array(scores2), np.array(y_dev_pred))

print(f'Mean Squared Error: {mse, mse2, mse3, mse4} \n Spearman: {res, res2, res3, res4} ')
#print(f'Mean Squared Error: {mse, mse2, mse3, mse4} \n Spearman: {res, res2, res3, res4} ')
print("-----------------")

# MLP reg
mlp_reg = MLPRegressor(random_state=42, max_iter=500).fit(X_train, y_train)
pred_tr = mlp_reg.predict(X_test)
pred_dev = mlp_reg.predict(merged_dev)
res_mlp1 = stats.spearmanr(np.array(y_test), np.array(pred_tr))
#res_mlp2 = stats.spearmanr(np.array(scores2), np.array(pred_dev))
#print(f"Scores for MLP: Train-{res_mlp1}, Dev-{res_mlp2}")
ensemble_model = VotingRegressor([('random_forest', rf_model), ('gradient_boosting', g_model)])

ensemble_model.fit(X_train, y_train)

ensemble_predictions = ensemble_model.predict(X_test)
ens_dev = ensemble_model.predict(merged_dev)
res_train_ens = stats.spearmanr(np.array(y_test), np.array(ensemble_predictions))
print(f"This is for ensemble:{res_train_ens}")
