# import libraries
import numpy as np
import pandas as pd
import os
from ast import literal_eval

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

data_path = 'data/'

city_df = pd.read_csv(os.path.join(data_path, 'city_population.csv'), sep=';', index_col='city')
city_df['population'] = city_df['population'].apply(lambda x: int(x.replace('\xa0', '')))
city_df = city_df.loc[~city_df.index.duplicated(keep='first')]


def get_city_population(line):
    if line in city_df.index:
        return city_df['population'].loc[line]
    else:
        return 23869 / 2


def preprocessing(df, idx):
    df['date_created'] = df['date_created'].apply(pd.to_datetime)
    df['year_created'] = df['date_created'].apply(lambda x: x.year)
    df['month_created'] = df['date_created'].apply(lambda x: x.month)
    df['day_created'] = df['date_created'].apply(lambda x: x.day)
    df['wd_created'] = df['date_created'].apply(pd.datetime.weekday)

    # Change boolean features to 1/0 int
    df['delivery_available'] = df['delivery_available'].apply(int)
    df['payment_available'] = df['payment_available'].apply(int)

    # Let's create feature of how many times in dataset one owner are selling items.
    owner_counts_train = df['owner_id'].value_counts()
    df['owner_count'] = df['owner_id'].apply(lambda x: owner_counts_train[x])

    # How many words is in desc_text. Which desctibes size of the description.
    df['desc_count'] = df['desc_text'].apply(lambda x: len(x.split()))

    df['city_population'] = df['city'].apply(get_city_population)

    # Let's find how many item properties was added.
    df['properties_len'] = df['properties'].apply(lambda x: len(literal_eval(x)))

    min_max_features = ['category_id', 'subcategory_id', 'month_created', 'day_created', 'city']

    for feature in min_max_features:
        min_dict = {}
        max_dict = {}

        for i in df[feature].unique():
            feature_prices = df[df[feature] == i]['price']

            min_dict[i] = feature_prices.min()
            max_dict[i] = feature_prices.max()

        df['price_' + feature + '_def'] = df.apply(lambda x: (x['price'] - min_dict[x[feature]]) / max_dict[x[feature]],
                                                   axis=1)

    chosen_cols = [
        'category_id',
        'city',
        'delivery_available',
        'img_num',
        'lat',
        'long',
        'payment_available',
        'price',
        'product_type',
        'sold_mode',
        'subcategory_id',
        'month_created',
        'wd_created',
        'day_created',
        'owner_count',
        'desc_count',
        'city_population',
        'properties_len',
        'price_category_id_def',
        'price_month_created_def',
        'price_day_created_def'
    ]

    df = df[chosen_cols]

    categorical_cols = [
        'city'
    ]

    # Features that we will transform with OHE
    dummy_cols = [
        'category_id',
        'product_type',
        'subcategory_id',
        'month_created',
        'wd_created',
        'day_created',
    ]

    df = pd.get_dummies(df, columns=dummy_cols)

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print('Preprocessed')

    return df[:idx], df[idx:]


train_df = pd.read_csv(os.path.join(data_path, 'train.tsv.gz'), sep='\t', index_col='product_id')
test_df = pd.read_csv(os.path.join(data_path, 'test_nolabel.tsv.gz'), sep='\t', index_col='product_id')

y_train = train_df['sold_fast']
train_df.drop(['sold_fast'], axis=1, inplace=True)

# Firstly, let's remove outliers
target = y_train[train_df['price'] < train_df['price'].max() * 0.005]
train_df = train_df[train_df['price'] < train_df['price'].max() * 0.005]

# Concatenate train_df and test_df.
train_test_df = pd.concat([train_df, test_df])

X_train, X_test = preprocessing(train_test_df, train_df.shape[0])

# Prediction

param = {
    'bagging_freq': 8,
    'bagging_fraction': 0.6,
    'boost_from_average': 'false',
    'boost': 'gbdt',
    'feature_fraction': 0.09,
    'learning_rate': 0.0075,
    'max_depth': -1,
    'metric': 'auc',
    'min_data_in_leaf': 50,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 8,
    'num_threads': 6,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1
}

np.random.seed(1)

lgb_x_train = lgb.Dataset(X_train, label=target)
clf = lgb.train(param, lgb_x_train, 40000, verbose_eval=10)
lgb_test_pred = clf.predict(X_test.values, num_iteration=clf.best_iteration)

df_submission = pd.DataFrame({'score': lgb_test_pred},
                             index=test_df.index)

submission_filename = 'final_submit.csv'
df_submission.to_csv(submission_filename)
print('Submission saved to {}'.format(submission_filename))
