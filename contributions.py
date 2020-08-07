import pandas as pd
import numpy as np


def get_variation(df, percentiles=10):
    varis = {}
    for col in df.columns:
        if df.dtypes[col].name == 'category':
            varis[col] = list(df[col].unique())
            continue
        else:
            varis[col] = [df[col].quantile(i) for i in np.linspace(1 / percentiles, 1, percentiles)]

    return varis


def argsort(df, orig):
    values = df.values
    orig_values = orig.values
    cols = df.columns
    ans = []
    for n in range(len(values)):
        #         ans.append([cols[i] + ' = ' + str(orig_values[n, i]) + ', ' + str(values[n, i]) for i in np.argsort(-abs(values[n]))])
        ans.append(cols[np.argsort(-abs(values[n]))])
    #         ans.append([cols[i] + ' = {:.2f}'.format(values[n, i]) for i in np.argsort(-abs(values[n]))])

    ans = pd.DataFrame(np.stack(ans, axis=1).transpose())
    ans = ans.set_index(df.index)
    return ans


def batch_contr(model, batch_params, batch_labels, var_dict):
    ans = batch_params.copy()
    for col in batch_params.columns:
        variabs = var_dict[col]
        l = len(variabs)
        mid_df = pd.concat([batch_params] * l).reset_index(drop=True)
        mid_df[col] = np.stack([variabs] * len(batch_params)).transpose((1, 0)).reshape(-1)
        mid_preds = model.predict_proba(mid_df)
        mid_preds = np.reshape([i[1] for i in mid_preds], (l, -1)).mean(axis=0)
        #         mid_preds = model.predict(mid_df)
        #         mid_preds = np.reshape(mid_preds, (l, -1)).mean(axis=0)

        ans[col] = mid_preds - batch_labels

    return ans


def get_contribution(model, df, target=None, batch_size=1000, target_df=None, calculate=False):
    if calculate:
        params = df
        labels = model.predict_proba(df)
        labels = [i[1] for i in labels]
    #         labels = model.predict(df)
    elif target_df is not None:
        params = df
        labels = target_df.values
    else:
        if not target:
            target = df.columns[-1]
        params = df.drop(target, axis=1)
        labels = df.target.values

    var_dict = get_variation(df, percentiles=50)

    ans = []

    batch_num = (len(df) - 1) // batch_size + 1
    for i in range(batch_num):
        variations = batch_contr(model,
                                 params.iloc[i * batch_size: min((i + 1) * batch_size, len(df)), :],
                                 labels[i * batch_size: min((i + 1) * batch_size, len(df))],
                                 var_dict)
        ans.append(variations)
    #         print(variations.head())

    ans = pd.concat(ans)
    ans = argsort(ans, params)

    return ans
