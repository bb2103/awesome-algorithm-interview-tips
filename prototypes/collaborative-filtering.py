import numpy as np
import pandas as pd

def cosine_similarity(vec1, vec2, normalize=False):
    ans = np.dot(vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2))
    if normalize:
        ans = 0.5 + 0.5 * ans
    return ans

def item_cf():
    userID = 1
    itemID = 'E'
    n = 2 # SELECT TOP-K Item Similarity
    # 1. 构建<商品、用户、评分> 矩阵
    items = {'A': {1: 5, 2: 3, 3: 4, 4: 3, 5: 1},
             'B': {1: 3, 2: 1, 3: 3, 4: 3, 5: 5},
             'C': {1: 4, 2: 2, 3: 4, 4: 1, 5: 5},
             'D': {1: 4, 2: 3, 3: 3, 4: 5, 5: 2},
             'E': {2: 3, 3: 5, 4: 4, 5: 1}}
    item_df = pd.DataFrame(items).T
    print(item_df)

    # 2. 计算物品相似度
    similarity_df = pd.DataFrame(np.ones((len(items), len(items))), index=['A', 'B', 'C', 'D', 'E'], columns=['A', 'B', 'C', 'D', 'E'])
    for i in range(item_df.shape[0]):
        for j in range(i, item_df.shape[0]):
            # Direct Get item_df.values[i], NAN Will Lead NAN Result, So Fill NA WITH ZERO
            similarity = cosine_similarity(item_df.iloc[i].fillna(0.), item_df.iloc[j].fillna(0.), normalize=True)
            similarity_df.values[i][j] = similarity
            similarity_df.values[j][i] = similarity
    print(similarity_df)

    # 3. 计算最相似的Top-K个物品
    similarity_items = similarity_df[itemID].sort_values(ascending=False)[:n+1].index.tolist()[1:]
    print("Item %s Search Top-%d Similarity Items: %s" % (itemID, n, similarity_items))

    # 4. 基于相似的商品推荐物品的最终得分 s = item_avg + \sum_{品}{品相似度 * (用户对商品的评分 - 品平均分) / (品相似度总和)}
    item_bias = np.mean(item_df.loc[itemID]) # NAN WITH SKIP
    weighted_sum = 0
    for item in similarity_items:
        bias = item_df.loc[item].mean() # 每个品的平均值
        weighted_sum += (item_df.loc[item, userID] - bias) * similarity_df.loc[itemID, item]
    topk_item_similarity_sum = similarity_df.loc[itemID, similarity_items].sum()
    pred = item_bias + weighted_sum / topk_item_similarity_sum
    item_df.loc[itemID, userID] = pred
    print(pred)
    print(item_df)

def user_cf():
    userID = 1
    itemID = 'E'
    n = 2 # SELECT TOP-K User Similarity

    # 1. 构建<用户,商品,评分>表
    users = {1: {'A': 5, 'B': 3, 'C': 4, 'D': 4}, # NOTICE: Predict E
             2: {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
             3: {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
             4: {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
             5: {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}}
    user_df = pd.DataFrame(users).T # DF为列存
    print(user_df)

    # 2. 计算用户相似度
    similarity_df = pd.DataFrame(np.zeros((len(users), len(users))), index=user_df.index, columns=user_df.index)
    for i in range(user_df.shape[0]):
        for j in range(i, user_df.shape[0]):
            # Direct Get user_df.values[i], NAN Will Lead NAN Result, So Fill NA WITH ZERO
            similarity = cosine_similarity(user_df.iloc[i].fillna(0.), user_df.iloc[j].fillna(0.), normalize=True)
            similarity_df.values[i][j] = similarity
            similarity_df.values[j][i] = similarity
    print(similarity_df)

    # 3.计算前n个相似的用户
    similarity_users = similarity_df[userID].sort_values(ascending=False)[:n+1].index.tolist()[1:]
    print("User %s Search Top-%d Similarity Users: %s" % (userID, n, similarity_users))

    # 4. 基于相似的用户推荐物品的最终得分 s = user_avg + \sum_{用户}{用户相似度 * (用户对商品的评分 - 用户平均分) / (用户相似度总和)}
    base_score = np.mean(np.array([value for value in users[userID].values()]))
    weighted_scores = 0.
    corr_values_sum = 0.
    for user in similarity_users:
        corr_value = similarity_df[userID][user]            # 两个用户之间的相似性
        mean_user_score = np.mean(np.array([value for value in users[user].values()]))    # 每个用户的打分平均值
        weighted_scores += corr_value * (users[user][itemID] - mean_user_score)      # 加权分数
        corr_values_sum += corr_value
    final_scores = base_score + weighted_scores / corr_values_sum
    print('用户%d对物品E的打分: %f' % (userID, final_scores))
    user_df.loc[userID][itemID] = final_scores
    print(user_df)

item_cf()
user_cf()

# Related
# https://zhuanlan.zhihu.com/p/268401920

