import os
import pandas as pd
import numpy as np

from itertools import combinations
import collections


import warnings
warnings.filterwarnings("ignore")

PATH_TO_DATA = '../data/'


def read_matches(matches_file):
    
    MATCHES_COUNT = {
        'test_matches.jsonl': 10000,
        'train_matches.jsonl': 39675,
    }
    _, filename = os.path.split(matches_file)
    total_matches = MATCHES_COUNT.get(filename)
    
    with open(matches_file) as fin:
        for line in tqdm_notebook(fin, total=total_matches):
            yield json.loads(line)


try:
    import ujson as json
except ModuleNotFoundError:
    import json
    print ('Please install ujson to read JSON oblects faster')
    
try:
    from tqdm import tqdm_notebook
except ModuleNotFoundError:
    tqdm_notebook = lambda x: x
    print ('Please install tqdm to track progress with Python loops')

#==========================================================================================================
#
#    Generate Hero Items features
#
#==========================================================================================================


def extract_features_csv(match):
    
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]

    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)

        row.append( (f'{player_name}_items', list(map(lambda x: x['key'], player['purchase_log'])) ) )
        #here u can extract other data

    return collections.OrderedDict(row)



def extract_inverse_features_csv(match):
    
    row = [
        ('match_id_hash', match['match_id_hash'][::-1]),
    ]

    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'd%d' % (slot + 1)
        else:
            player_name = 'r%d' % (slot - 4)

        row.append( (f'{player_name}_items', list(map(lambda x: x['key'], player['purchase_log'])) ) )
        #here u can extract other data

    return collections.OrderedDict(row)



def create_features_from_jsonl(matches_file):
  
    df_new_features = []

    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']
        features = extract_features_csv(match)

        df_new_features.append(features)

    df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
    return df_new_features


def create_inverse_features_from_jsonl(matches_file):
  
    df_new_features = []

    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']
        features = extract_features_csv(match)
        inverse_features = extract_inverse_features_csv(match)

        df_new_features.append(features)
        df_new_features.append(inverse_features)

    df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
    return df_new_features



def add_items_dummies(train_df, test_df):
    
    full_df = pd.concat([train_df, test_df], sort=False)
    train_size = train_df.shape[0]

    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        item_columns = [f'{player}_items' for player in players]

        d = pd.get_dummies(full_df[item_columns[0]].apply(pd.Series).stack()).sum(level=0, axis=0)
        dindexes = d.index.values

        for c in item_columns[1:]:
            d = d.add(pd.get_dummies(full_df[c].apply(pd.Series).stack()).sum(level=0, axis=0), fill_value=0)
            d = d.ix[dindexes]

        full_df = pd.concat([full_df, d.add_prefix(f'{team}_item_')], axis=1, sort=False)
        full_df.drop(columns=item_columns, inplace=True)

    train_df = full_df.iloc[:train_size, :]
    test_df = full_df.iloc[train_size:, :]

    return train_df, test_df


def drop_consumble_items(train_df, test_df):
    
    full_df = pd.concat([train_df, test_df], sort=False)
    train_size = train_df.shape[0]

    for team in 'r', 'd':
        consumble_columns = ['tango', 'tpscroll', 
                             'bottle', 'flask',
                            'enchanted_mango', 'clarity',
                            'faerie_fire', 'ward_observer',
                            'ward_sentry']
        
        starts_with = f'{team}_item_'
        consumble_columns = [starts_with + column for column in consumble_columns]
        full_df.drop(columns=consumble_columns, inplace=True)

    train_df = full_df.iloc[:train_size, :]
    test_df = full_df.iloc[train_size:, :]

    return train_df, test_df


print('Generate Hero Items features')

train_df = create_inverse_features_from_jsonl(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')).fillna(0)
test_df = create_features_from_jsonl(os.path.join(PATH_TO_DATA, 'test_matches.jsonl')).fillna(0)

new_train, new_test = add_items_dummies(train_df, test_df)
new_train, new_test = drop_consumble_items(new_train, new_test)


import pickle as pkl

new_train.to_pickle('train_hero_items.pkl')
new_test.to_pickle('test_hero_items.pkl')


print('Done!!!')
print()

















#==========================================================================================================
#
#    Generate Main features
#
#==========================================================================================================




##train
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold #for cross-validation
from sklearn.metrics import roc_auc_score, log_loss #this is we are trying to increase
from itertools import combinations
import lightgbm as lgb 
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")
import math

import time
import datetime


            
def get_average_speed(arr_t, times):
  
    speed_sum = 0
    for i in range(1, len(times)):
        delta_arr = arr_t[i] - arr_t[i - 1]
        delta_time = times[i] - times[i - 1]
        speed_sum += delta_arr / delta_time
    
    average_speed = speed_sum / (len(times) - 1) if len(times) > 1 else 0
    
    return average_speed
    
import collections

MATCH_FEATURES = [
    ('game_time', lambda m: m['game_time']),
    ('game_mode', lambda m: m['game_mode']),
    ('lobby_type', lambda m: m['lobby_type']),
    ('objectives_len', lambda m: len(m['objectives'])),
    ('chat_len', lambda m: len(m['chat'])),
]

PLAYER_FIELDS = [
    'hero_id',
    
    'kills',
    'deaths',
    'assists',
    'denies',
    
    'gold',
    'lh',
    'xp',
    'health',
    'max_health',
    'max_mana',
    'level',

    'x',
    'y',
    
    'stuns',
    'creeps_stacked',
    'camps_stacked',
    'rune_pickups',
    'firstblood_claimed',
    'teamfight_participation',
    'roshans_killed',
    'obs_placed',
    'sen_placed',
]

def extract_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]
    
    for field, f in MATCH_FEATURES:
        row.append((field, f(match)))
        
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)

        for field in PLAYER_FIELDS:
            column_name = '%s_%s' % (player_name, field)
            row.append((column_name, player[field]))
        row.append((f'{player_name}_ability_level', len(player['ability_upgrades'])))
        row.append((f'{player_name}_max_hero_hit', player['max_hero_hit']['value']))
        row.append((f'{player_name}_purchase_count', len(player['purchase_log'])))
        row.append((f'{player_name}_count_ability_use', sum(player['ability_uses'].values())))
        row.append((f'{player_name}_damage_dealt', sum(player['damage'].values())))
        row.append((f'{player_name}_damage_received', sum(player['damage_taken'].values())))
        
        ##new
        row.append( (f'{player_name}_randomed', int(player['randomed'])))
        row.append((f'{player_name}_buyback', len(player['buyback_log'])))
        #row.append( (f'{player_name}_items', list(map(lambda x: x['id'][5:], player['hero_inventory'])) ) )
        row.append((f'{player_name}_life_state_1', player['life_state']['1'] if len(player['life_state']) > 2 else 0))
        row.append((f'{player_name}_life_state_2', player['life_state']['2'] if len(player['life_state']) > 2 else 0))
        #row.append((f'{player_name}_pred_win', int(player['pred_vict'])))
        row.append((f'{player_name}_gold_speed', get_average_speed(player['gold_t'], player['times'])))
        row.append((f'{player_name}_xp_speed', get_average_speed(player['xp_t'], player['times'])))
        row.append((f'{player_name}_lh_speed', get_average_speed(player['lh_t'], player['times'])))
        
        #gold_reasons
        for reason in ['0', '1', '5', '6', '11', '12', '13']:
            row.append((f'{player_name}_gold_reason_{reason}', player['gold_reasons'][reason] if reason in player['gold_reasons'].keys() else 0))
            
        #for reason in ['0', '1', '2']:
        #    row.append((f'{player_name}_xp_reason_{reason}', player['xp_reasons'][reason] if reason in player['xp_reasons'].keys() else 0))
        
        #for reason in range(2, 6):
            #row.append((f'{player_name}_multikill_{reason}', player['multi_kills'][str(reason)] if str(reason) in player['multi_kills'].keys() else 0))
        ##new
            
    return collections.OrderedDict(row)
  
    
    
def extract_inverse_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash'][::-1]),
    ]
    
    for field, f in MATCH_FEATURES:
        row.append((field, f(match)))
        
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'd%d' % (slot + 1)
        else:
            player_name = 'r%d' % (slot - 4)

        for field in PLAYER_FIELDS:
            column_name = '%s_%s' % (player_name, field)
            row.append((column_name, player[field]))
        row.append((f'{player_name}_ability_level', len(player['ability_upgrades'])))
        row.append((f'{player_name}_max_hero_hit', player['max_hero_hit']['value']))
        row.append((f'{player_name}_purchase_count', len(player['purchase_log'])))
        row.append((f'{player_name}_count_ability_use', sum(player['ability_uses'].values())))
        row.append((f'{player_name}_damage_dealt', sum(player['damage'].values())))
        row.append((f'{player_name}_damage_received', sum(player['damage_taken'].values())))
        
        ##new
        row.append( (f'{player_name}_randomed', int(player['randomed'])))
        row.append((f'{player_name}_buyback', len(player['buyback_log'])))
        #row.append( (f'{player_name}_items', list(map(lambda x: x['id'][5:], player['hero_inventory'])) ) )
        row.append((f'{player_name}_life_state_1', player['life_state']['1'] if len(player['life_state']) > 2 else 0))
        row.append((f'{player_name}_life_state_2', player['life_state']['2'] if len(player['life_state']) > 2 else 0))
        #row.append((f'{player_name}_pred_win', int(player['pred_vict'])))
        row.append((f'{player_name}_gold_speed', get_average_speed(player['gold_t'], player['times'])))
        row.append((f'{player_name}_xp_speed', get_average_speed(player['xp_t'], player['times'])))
        row.append((f'{player_name}_lh_speed', get_average_speed(player['lh_t'], player['times'])))
        
        #gold_reasons
        for reason in ['0', '1', '5', '6', '11', '12', '13']:
            row.append((f'{player_name}_gold_reason_{reason}', player['gold_reasons'][reason] if reason in player['gold_reasons'].keys() else 0))
            
        #for reason in ['0', '1', '2']:
        #    row.append((f'{player_name}_xp_reason_{reason}', player['xp_reasons'][reason] if reason in player['xp_reasons'].keys() else 0))
        
        #for reason in range(2, 6):
            #row.append((f'{player_name}_multikill_{reason}', player['multi_kills'][str(reason)] if str(reason) in player['multi_kills'].keys() else 0))
        ##new
            
    return collections.OrderedDict(row)
    
def extract_targets_csv(match, targets):
    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [
        (field, targets[field])
        for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']
    ])

def extract_inverse_targets_csv(match, targets):
    return collections.OrderedDict([('match_id_hash', match['match_id_hash'][::-1])] + [
        (field,  not targets[field])
        for field in ['radiant_win']])
    
def create_features_from_jsonl(matches_file):
  
    df_new_features = []
    
    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']
        features = extract_features_csv(match)
        
        df_new_features.append(features)

    df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
    return df_new_features


def create_inverse_features_from_jsonl(matches_file):
  
    df_new_features = []
    df_new_targets = []
    
    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']
        features = extract_features_csv(match)
        inverse_features = extract_inverse_features_csv(match)
        
        df_new_features.append(features)
        df_new_features.append(inverse_features)

        targets = extract_targets_csv(match, match['targets'])
        inverse_targets = extract_inverse_targets_csv(match, match['targets'])

        df_new_targets.append(targets)
        df_new_targets.append(inverse_targets)

    df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
    df_new_targets = pd.DataFrame.from_records(df_new_targets).set_index('match_id_hash')
    return df_new_features, df_new_targets

def add_advanced_features_fron_jsonl(df_features, matches_file):
    
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']

        # Counting ruined towers for both teams
        radiant_tower_kills = 0
        dire_tower_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 2:
                    radiant_tower_kills += 1
                if objective['team'] == 3:
                    dire_tower_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'r_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'd_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills        
        df_features.loc[match_id_hash, 'ratio_tower_kills'] = (radiant_tower_kills + 0.01) / (0.01 + dire_tower_kills)
        # ... here you can add more features ...
        
        # BARAQS
        radiant_baraq_kills = 0
        dire_baraq_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_BARRACKS_KILL':
                if int(objective['key']) in [64,128,256,512,1024,2048]:
                    radiant_baraq_kills += 1
                if int(objective['key']) in [1,2,4,8,16,32]:
                    dire_baraq_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'r_baraq_kills'] = radiant_baraq_kills
        df_features.loc[match_id_hash, 'd_baraq_kills'] = dire_baraq_kills
        df_features.loc[match_id_hash, 'diff_baraq_kills'] = radiant_baraq_kills - dire_baraq_kills
        df_features.loc[match_id_hash, 'ratio_baraq_kills'] = radiant_baraq_kills / (0.01 + dire_baraq_kills)
        
        # Total damage
        total_damage = 0
        for i in range(1, 6):
            for j in match['players'][i-1]['damage']:
                # Take damage only to hero(not for creeps)
                if j.startswith('npc_dota_hero'):
                    total_damage += match['players'][i-1]['damage'][j]
        df_features.loc[match_id_hash, 'r_champ_damage'] = total_damage
        
        total_damage = 0
        for i in range(6, 11):
            for j in match['players'][i-1]['damage']:
                if j.startswith('npc_dota_hero'):
                    total_damage += match['players'][i-1]['damage'][j]
        df_features.loc[match_id_hash, 'd_champ_damage'] = total_damage

        df_features.loc[match_id_hash, 'diff_champ_damage'] = df_features.loc[match_id_hash, 'r_champ_damage'] - df_features.loc[match_id_hash, 'd_champ_damage'] 
        

def add_inverse_advanced_features_fron_jsonl(df_features, matches_file):
    
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']

        # Counting ruined towers for both teams
        radiant_tower_kills = 0
        dire_tower_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 2:
                    radiant_tower_kills += 1
                if objective['team'] == 3:
                    dire_tower_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'r_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'd_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills        
        df_features.loc[match_id_hash, 'ratio_tower_kills'] = (radiant_tower_kills + 0.01) / (0.01 + dire_tower_kills)
        # ... here you can add more features ...
        
        # BARAQS
        radiant_baraq_kills = 0
        dire_baraq_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_BARRACKS_KILL':
                if int(objective['key']) in [64,128,256,512,1024,2048]:
                    radiant_baraq_kills += 1
                if int(objective['key']) in [1,2,4,8,16,32]:
                    dire_baraq_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'r_baraq_kills'] = radiant_baraq_kills
        df_features.loc[match_id_hash, 'd_baraq_kills'] = dire_baraq_kills
        df_features.loc[match_id_hash, 'diff_baraq_kills'] = radiant_baraq_kills - dire_baraq_kills
        df_features.loc[match_id_hash, 'ratio_baraq_kills'] = radiant_baraq_kills / (0.01 + dire_baraq_kills)
        
        # Total damage
        total_damage = 0
        for i in range(1, 6):
            for j in match['players'][i-1]['damage']:
                # Take damage only to hero(not for creeps)
                if j.startswith('npc_dota_hero'):
                    total_damage += match['players'][i-1]['damage'][j]
        df_features.loc[match_id_hash, 'r_champ_damage'] = total_damage
        
        total_damage = 0
        for i in range(6, 11):
            for j in match['players'][i-1]['damage']:
                if j.startswith('npc_dota_hero'):
                    total_damage += match['players'][i-1]['damage'][j]
        df_features.loc[match_id_hash, 'd_champ_damage'] = total_damage

        df_features.loc[match_id_hash, 'diff_champ_damage'] = df_features.loc[match_id_hash, 'r_champ_damage'] - df_features.loc[match_id_hash, 'd_champ_damage'] 
        
        
        
        #### Inverted data!!!
        match_id_hash = match['match_id_hash'][::-1]

        # Counting ruined towers for both teams
        radiant_tower_kills = 0
        dire_tower_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
                if objective['team'] == 3:
                    radiant_tower_kills += 1
                if objective['team'] == 2:
                    dire_tower_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'r_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'd_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills        
        df_features.loc[match_id_hash, 'ratio_tower_kills'] = (radiant_tower_kills + 0.01) / (0.01 + dire_tower_kills)
        # ... here you can add more features ...
        
        # BARAQS
        radiant_baraq_kills = 0
        dire_baraq_kills = 0
        for objective in match['objectives']:
            if objective['type'] == 'CHAT_MESSAGE_BARRACKS_KILL':
                if int(objective['key']) in [64,128,256,512,1024,2048]:
                    dire_baraq_kills += 1
                if int(objective['key']) in [1,2,4,8,16,32]:
                    radiant_baraq_kills += 1

        # Write new features
        df_features.loc[match_id_hash, 'r_baraq_kills'] = radiant_baraq_kills
        df_features.loc[match_id_hash, 'd_baraq_kills'] = dire_baraq_kills
        df_features.loc[match_id_hash, 'diff_baraq_kills'] = radiant_baraq_kills - dire_baraq_kills
        df_features.loc[match_id_hash, 'ratio_baraq_kills'] = radiant_baraq_kills / (0.01 + dire_baraq_kills)
        
        # Total damage
        total_damage = 0
        for i in range(1, 6):
            for j in match['players'][i-1]['damage']:
                # Take damage only to hero(not for creeps)
                if j.startswith('npc_dota_hero'):
                    total_damage += match['players'][i-1]['damage'][j]
        df_features.loc[match_id_hash, 'd_champ_damage'] = total_damage
        
        total_damage = 0
        for i in range(6, 11):
            for j in match['players'][i-1]['damage']:
                if j.startswith('npc_dota_hero'):
                    total_damage += match['players'][i-1]['damage'][j]
        df_features.loc[match_id_hash, 'r_champ_damage'] = total_damage

        df_features.loc[match_id_hash, 'diff_champ_damage'] = df_features.loc[match_id_hash, 'r_champ_damage'] - df_features.loc[match_id_hash, 'd_champ_damage'] 
        

print('Generate Main features')

df_train_new_features, df_new_targets = create_inverse_features_from_jsonl(os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))
df_test_new_features = create_features_from_jsonl(os.path.join(PATH_TO_DATA, 'test_matches.jsonl')).fillna(0)
df_train_new_features = df_train_new_features.fillna(0)

print('Done!!!')


print('Generate advanced features')
add_inverse_advanced_features_fron_jsonl(df_train_new_features, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))
add_advanced_features_fron_jsonl(df_test_new_features, os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))

print ('Done!!!')

df_train_new_features.to_pickle('df_train_features_inverted.pkl')
df_test_new_features.to_pickle('df_test_features_inverted.pkl')
df_new_targets.to_csv('new_inverse_targets.csv')




#==========================================================================================================
#
#    Update features
#
#==========================================================================================================






def add_new_features_from_df(train, test, feature_array, delete_list):

    for c in delete_list:
        r_columns = [f'r{i}_{c}' for i in range(1, 6)]
        d_columns = [f'd{i}_{c}' for i in range(1, 6)]
        
        train.drop(columns = (r_columns + d_columns), axis=1, inplace=True)
        test.drop(columns = (r_columns + d_columns), axis=1, inplace=True)
    
    for c in feature_array:

        r_columns = [f'r{i}_{c}' for i in range(1, 6)]
        d_columns = [f'd{i}_{c}' for i in range(1, 6)]
        
        # TOTAL
        train['r_total_' + c] = train[r_columns].sum(1)
        train['d_total_' + c] = train[d_columns].sum(1)
        train['total_' + c + '_ratio'] = (train['r_total_' + c] + 0.01) / (0.01 + train['d_total_' + c])
        train['total_' + c + '_diff'] = train['r_total_' + c] - train['d_total_' + c]
        #train.drop(['r_total_' + c, 'd_total_' + c], axis=1, inplace=True)

        test['r_total_' + c] = test[r_columns].sum(1)
        test['d_total_' + c] = test[d_columns].sum(1)
        test['total_' + c + '_ratio'] = (test['r_total_' + c] + 0.01) / (0.01 + test['d_total_' + c])
        test['total_' + c + '_diff'] = test['r_total_' + c] - test['d_total_' + c]
        #test.drop(['r_total_' + c, 'd_total_' + c], axis=1, inplace=True)

        # STD
        #train['r_std_' + c] = train[r_columns].std(1)
        #train['d_std_' + c] = train[d_columns].std(1)
        #train['std_' + c + '_ratio'] = (train['r_std_' + c] + 0.01) / (0.01 + train['d_std_' + c])
        #train['std_' + c + '_diff'] = train['r_std_' + c] - train['d_std_' + c]
        #train.drop(['r_std_' + c, 'd_std_' + c], axis=1, inplace=True)

        #test['r_std_' + c] = test[r_columns].std(1)
        #test['d_std_' + c] = test[d_columns].std(1)
        #test['std_' + c + '_ratio'] = (test['r_std_' + c] + 0.01) / (0.01 + test['d_std_' + c])
        #test['std_' + c + '_diff'] = test['r_std_' + c] - test['d_std_' + c]
        #test.drop(['r_std_' + c, 'd_std_' + c], axis=1, inplace=True)

        # MEAN
        #train['r_mean_' + c] = train[r_columns].mean(1)
        #train['d_mean_' + c] = train[d_columns].mean(1)
        #train['mean_' + c + '_ratio'] = (train['r_mean_' + c] + 0.01) / (0.01 + train['d_mean_' + c])
        #train['mean_' + c + '_diff'] = train['r_mean_' + c] - train['d_mean_' + c]
        #train.drop(['r_mean_' + c, 'd_mean_' + c], axis=1, inplace=True)

        #test['r_mean_' + c] = test[r_columns].mean(1)
        #test['d_mean_' + c] = test[d_columns].mean(1)
        #test['mean_' + c + '_ratio'] = (test['r_mean_' + c] + 0.01) / (0.01 + test['d_mean_' + c])
        #test['mean_' + c + '_diff'] = test['r_mean_' + c] - test['d_mean_' + c]
        #test.drop(['r_mean_' + c, 'd_mean_' + c], axis=1, inplace=True)
        
        #MIN
        train['r_min_' + c] = train[r_columns].min(1)
        train['d_min_' + c] = train[d_columns].min(1)
        #train['min_' + c + '_ratio'] = (train['r_min_' + c] + 0.01) / (0.01 + train['d_min_' + c])
        #train['min_' + c + '_diff'] = train['r_min_' + c] - train['d_min_' + c]

        test['r_min_' + c] = test[r_columns].min(1)
        test['d_min_' + c] = test[d_columns].min(1)
        #test['min_' + c + '_ratio'] = (test['r_min_' + c] + 0.01) / (0.01 + test['d_min_' + c])
        #test['min_' + c + '_diff'] = test['r_min_' + c] - test['d_min_' + c]
        
        #MAX
        train['r_max_' + c] = train[r_columns].min(1)
        train['d_max_' + c] = train[d_columns].min(1)
        #train['min_' + c + '_ratio'] = (train['r_min_' + c] + 0.01) / (0.01 + train['d_min_' + c])
        #train['min_' + c + '_diff'] = train['r_min_' + c] - train['d_min_' + c]

        test['r_max_' + c] = test[r_columns].min(1)
        test['d_max_' + c] = test[d_columns].min(1)
        #test['max_' + c + '_ratio'] = (test['r_max_' + c] + 0.01) / (0.01 + test['d_max_' + c])
        test['max_' + c + '_diff'] = test['r_max_' + c] - test['d_max_' + c]

        train.drop(labels = (r_columns + d_columns), axis=1, inplace=True)
        test.drop(labels = (r_columns + d_columns), axis=1, inplace=True)

    return train, test
    
def add_hero_dummies(train_df, test_df):
  
    full_df = pd.concat([train_df, test_df], sort=False)
    train_size = train_df.shape[0]

    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        hero_columns = [f'{player}_hero_id' for player in players]
        d = pd.get_dummies(full_df[hero_columns[0]])

        for c in hero_columns[1:]:
            d += pd.get_dummies(full_df[c])

        full_df = pd.concat([full_df, d.add_prefix(f'{team}_hero_')], axis=1, sort=False)
        full_df.drop(columns=hero_columns, inplace=True)

    train_df = full_df.iloc[:train_size, :]
    test_df = full_df.iloc[train_size:, :]

    return train_df, test_df
    
def add_randomed(train_df, test_df):
  
    full_df = pd.concat([train_df, test_df], sort=False)
    train_size = train_df.shape[0]

    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        random_columns = [f'{player}_randomed' for player in players]
        d = full_df[random_columns[0]]

        for c in random_columns[1:]:
            d += full_df[c] 
        
        full_df[f'{team}_randomed'] = d
        full_df.drop(columns=random_columns, inplace=True)

    train_df = full_df.iloc[:train_size, :]
    test_df = full_df.iloc[train_size:, :]

    return train_df, test_df

def create_new_features(train_df, test_df):
  
    new_train = train_df.copy()
    new_test = test_df.copy()
      
    new_train['r_damage_dealt_to_minions'] = 0
    new_train['d_damage_dealt_to_minions'] = 0
    new_test['r_damage_dealt_to_minions'] = 0
    new_test['d_damage_dealt_to_minions'] = 0
      
    for i in range(1,6):
        new_train[f'r_damage_dealt_to_minions'] += new_train[f'r{i}_damage_dealt']
        new_train[f'd_damage_dealt_to_minions'] += new_train[f'd{i}_damage_dealt'] 
        
        new_test[f'r_damage_dealt_to_minions'] += new_test[f'r{i}_damage_dealt']
        new_test[f'd_damage_dealt_to_minions'] += new_test[f'd{i}_damage_dealt']
        
    new_train['r_damage_dealt_to_minions'] -= new_train['r_champ_damage']
    new_train['d_damage_dealt_to_minions'] -= new_train['d_champ_damage']
      
    new_test['r_damage_dealt_to_minions'] -= new_test['r_champ_damage']
    new_test['d_damage_dealt_to_minions'] -= new_test['d_champ_damage']
      
    ###
    new_train['min_damage_ratio'] = new_train['r_damage_dealt_to_minions'] / (0.01 + new_train['d_damage_dealt_to_minions'])
    new_test['min_damage_ratio'] = new_test['r_damage_dealt_to_minions'] / (0.01 +  new_test['d_damage_dealt_to_minions'])
    
    # MEAN
    new_train['r_mean_min_damage'] = new_train['r_damage_dealt_to_minions'] / 5
    new_train['d_mean_min_damage'] = new_train['r_damage_dealt_to_minions'] / 5
    new_train['mean_min_damage_ratio'] = new_train['r_mean_min_damage'] / (0.01 + new_train['d_mean_min_damage'])
    new_train.drop(['r_mean_min_damage', 'd_mean_min_damage'], axis=1, inplace=True)
    
    new_test['r_mean_min_damage'] = new_test['r_damage_dealt_to_minions'] / 5
    new_test['d_mean_min_damage'] = new_test['r_damage_dealt_to_minions'] / 5
    new_test['mean_min_damage_ratio'] = new_test['r_mean_min_damage'] / (0.01 + new_test['d_mean_min_damage'])
    new_test.drop(['r_mean_min_damage', 'd_mean_min_damage'], axis=1, inplace=True)
      
    new_train.drop(columns = ['r_champ_damage', 'd_champ_damage'], inplace=True)
    new_test.drop(columns = ['r_champ_damage', 'd_champ_damage'], inplace=True)
      
      
    return new_train, new_test
    


df_train_targets = pd.read_csv('new_inverse_targets.csv', index_col='match_id_hash')
y = df_train_targets['radiant_win']

target = pd.DataFrame(y)

def create_next_roshan_team_feature(train_df, test_df):
  
    y = df_train_targets['next_roshan_team'].fillna('null').replace({'null' : 0, 'Radiant' : 1, 'Dire' : 2}).values
    target = pd.DataFrame(y)

    new_train = train_df.copy()
    new_test = test_df.copy()
    
    param = {
        'bagging_freq': 5,  #handling overfitting
        'bagging_fraction': 0.5,  #handling overfitting - adding some noise
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.05, #handling overfitting
        'learning_rate': 0.01,  #the changes between one auc and a better one gets really small thus a small learning rate performs better
        'max_depth': -1,  
        'metric': 'multi_logloss',
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 10,
        'num_threads': 5,
        'tree_learner': 'serial',
        'objective': 'multiclass', 
        'num_class': 3,
        'verbosity': 1
    }

    #divide training data into train and validaton folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

    #placeholder for out-of-fold, i.e. validation scores
    oof = np.zeros((len(new_train), 3))

    #for predictions
    predictions = np.zeros((len(new_test), 3))

    #and for feature importance
    feature_importance_df = pd.DataFrame()

    #RUN THE LOOP OVER FOLDS
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(new_train.values, target.values)):

        X_train, y_train = new_train.iloc[trn_idx], target.iloc[trn_idx]
        X_valid, y_valid = new_train.iloc[val_idx], target.iloc[val_idx]

        print("Computing Fold {}".format(fold_))
        trn_data = lgb.Dataset(X_train, label = y_train)
        val_data = lgb.Dataset(X_valid, label = y_valid)

        num_round = 5000 
        verbose = 1000 
        stop = 500 

        #TRAIN THE MODEL
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=verbose, early_stopping_rounds = stop)
        
        #CALCULATE PREDICTION FOR VALIDATION SET
        oof[val_idx] = clf.predict(new_train.iloc[val_idx], num_iteration=clf.best_iteration)

        #CALCULATE PREDICTIONS FOR TEST DATA, using best_iteration on the fold
        predictions += clf.predict(new_test, num_iteration=clf.best_iteration) / folds.n_splits

    #print overall cross-validatino score
    score = log_loss(target, oof)
    print("CV score: {:<8.5f}".format(score))
    
    return oof, predictions
    
FEATURE_ARRAY = ['kills', 'deaths', 'assists', 'denies', 'gold', 'lh', 'xp', 'health', 'max_health', 'max_mana', 'level', 'x', 'y', 'stuns', 'creeps_stacked', 'camps_stacked', 'rune_pickups',
          'firstblood_claimed', 'teamfight_participation', 'roshans_killed', 'obs_placed', 'sen_placed', 'ability_level', 'max_hero_hit', 'purchase_count',
          'count_ability_use', 'damage_dealt', 'damage_received', 'buyback', 'life_state_1', 'life_state_2', 'gold_reason_0', 'gold_reason_1', 'gold_reason_11',
                'gold_reason_12', 'gold_reason_13', 'gold_reason_5', 'gold_reason_6', 'gold_speed', 'xp_speed', 'lh_speed']

BEST_FEATURES = ['rune_pickups', 'gold', 'lh', 'xp', 'health', 'max_mana', 'purchase_count', 'kills', 
                 'teamfight_participation', 'assists', 'x', 'y', 'buyback', 'creeps_stacked',
                 'gold_reason_5','gold_reason_6', 'gold_speed']

new_train = pd.read_pickle('df_train_features_inverted.pkl')
new_test = pd.read_pickle('df_test_features_inverted.pkl')

new_train.fillna(value=0, inplace=True)
new_test.fillna(value=0, inplace=True)


target = pd.DataFrame(y)

print('Update features...')
new_train, new_test = create_new_features(new_train, new_test)
new_train, new_test = add_new_features_from_df(new_train, new_test, BEST_FEATURES, list(set(FEATURE_ARRAY) - set(BEST_FEATURES)))


new_train, new_test = add_hero_dummies(new_train, new_test)
new_train, new_test = add_randomed(new_train, new_test)

bad_features = ['total_rune_pickups_diff', 'total_rune_pickups_ratio', 
                'total_lh_ratio', 'total_lh_diff', 'total_y_diff', 
                'total_y_ratio', 'min_damage_ratio', 
                'total_gold_speed_diff', 'total_gold_speed_ratio', 'ratio_tower_kills']

new_train.drop(columns= bad_features, inplace = True)
new_test.drop(columns= bad_features, inplace = True)
print('Done!!!')

print('Calculate NextRoshaTeam predictions...')
train_roshan, test_roshan = create_next_roshan_team_feature(new_train, new_test)
print('Done!!!')

roshan_test_df = pd.DataFrame(data = test_roshan, index=new_test.index, columns=['next_roshan_team_0', 'next_roshan_team_1', 'next_roshan_team_2'])
new_test = pd.concat([new_test, roshan_test_df], axis=1)

roshan_train_df = pd.DataFrame(data = train_roshan, index=new_train.index, columns=['next_roshan_team_0', 'next_roshan_team_1', 'next_roshan_team_2'])
new_train = pd.concat([new_train, roshan_train_df], axis=1)

train_hero_items = pd.read_pickle('train_hero_items.pkl')
test_hero_items = pd.read_pickle('test_hero_items.pkl')

new_train = pd.concat([new_train, train_hero_items], axis=1)
new_test = pd.concat([new_test, test_hero_items], axis=1)

import lightgbm as lgb
import numpy as np


param = {
        'bagging_freq': 8,
        'bagging_fraction': 0.6,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.09,
        'learning_rate': 0.0075,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 8,
        'num_threads': 6,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1
    }

 

np.random.seed(1)

lgb_x_train = lgb.Dataset(new_train.values, label=target)

# Cross-validate
#cv_results = lgb.cv(param, lgb_x_train, num_boost_round=20000, nfold=5, 
#                    verbose_eval=100, early_stopping_rounds=500)

#print('Current parameters:\n', param)
#print('\nBest num_boost_round:', len(cv_results['auc-mean']))
#print('Best CV score:', cv_results['auc-mean'][-1])

print('Train main model...')
clf = lgb.train(param, lgb_x_train, 14700, verbose_eval=10)
print('Done!!!')

lgb_test_pred = clf.predict(new_test.values)
lgb_train_pred = clf.predict(new_train.values)

df_submission_extended = pd.DataFrame(
    {'radiant_win_prob': lgb_test_pred}, 
    index=new_test.index,
)
df_submission_extended.to_csv('submission_2.csv')
  