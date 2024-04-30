from tqdm import tqdm
from pm4py.objects.petri_net.semantics import ClassicSemantics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import copy
import pandas as pd
import numpy as np
import random
import utils.data_aware_spn_utils as dspn_utils
import utils.activities_data_aware_spn_utils as adspn_utils
from utils.data_alignments import extract
from utils.general_utils import get_cat_columns_maximum_values_onehot

def split_log_train_test(log, train_perc):
    case_ids = log['case:concept:name'].drop_duplicates().tolist()
    random.shuffle(case_ids)
    train_log = log[log['case:concept:name'].isin(case_ids[:int(np.round(train_perc*len(case_ids)))])]
    test_log = log[log['case:concept:name'].isin(case_ids[int(np.round(train_perc*len(case_ids))):])]
    return train_log, test_log

def get_training_sets(observation_points, categorical_attrs, max_one_hot=100):
    training_sets = {trans: pd.DataFrame.from_dict(observation_points[trans]) for trans in observation_points}
    data_considered = set()
    for transition in tqdm(training_sets):
        dataset = copy.copy(training_sets[transition])
        dataset['target'] = dataset['fired'] == transition
        # breakpoint()
        attributes_to_test = set(categorical_attrs).intersection(dataset.columns)
        columns_accepted_for_one_hot = get_cat_columns_maximum_values_onehot(dataset, attributes_to_test, max_one_hot)
        columns_to_drop = list(set(attributes_to_test).difference(set(columns_accepted_for_one_hot)))
        dataset = dataset.drop(columns=columns_to_drop)
        data_considered = data_considered.union(set(dataset.columns[~dataset.columns.isin(['target', 'fired'])]))
        training_sets[transition] = pd.get_dummies(dataset.drop(columns=['fired']), columns=columns_accepted_for_one_hot).fillna(0)
        # breakpoint()
    return training_sets, data_considered

def always_one():
    return 1

def always_zero():
    return 0

def fit_classifiers(training_sets):
    classifiers = {}
    for transition in training_sets:
        # breakpoint()
        dataset = training_sets[transition]
        features = dataset.drop(columns=['target'])
        targets = dataset['target']
        if len(targets.unique()) == 2:
            logistic = LogisticRegression(random_state=0)
            clf = Pipeline([('scaler', StandardScaler()), ('logistic', logistic)]) 
            clf.fit(features, targets)
        elif dataset['target'].unique().tolist()[0] == 1:
            clf = always_one
        elif dataset['target'].unique().tolist()[0] == 0:
            clf = always_zero
        else:
            raise ValueError('Target must be boolean.')
        classifiers[transition] = copy.deepcopy(clf)
    return classifiers

def get_probability_from_occurrences(transitions, log):
    act_stats = log[['case:concept:name', 'concept:name']].rename(columns={'case:concept:name': 'counter'}).groupby('concept:name').count()
    act_stats = act_stats.reset_index()
    act_stats['prob'] = act_stats['counter'].copy()/len(log)
    trans_prob = {}
    for trans in transitions:
        trans_prob[trans] = act_stats[trans]
    return copy.copy(trans_prob)

def train(log, net, im, fm, categorical_attrs, not_data_attrs={'case:concept:name', 'time:timestamp', 'concept:name'}, semantic=ClassicSemantics()):
    data_al, log_model_align_map = extract(log, net, im, fm, not_data_attrs=not_data_attrs)
    # breakpoint()
    obs_points = adspn_utils.get_observation_points(data_al, log_model_align_map, net, im, semantic)
    training_sets, data_considered = get_training_sets(obs_points, categorical_attrs)
    # breakpoint()
    data_considered = data_considered.difference(set(log['concept:name'].unique().tolist()))
    return  fit_classifiers(training_sets), obs_points, training_sets, data_considered

def train_data_aware(log, net, im, fm, categorical_attrs, not_data_attrs={'case:concept:name', 'time:timestamp', 'concept:name'}, semantic=ClassicSemantics()):
    data_al, log_model_align_map = extract(log, net, im, fm, not_data_attrs=not_data_attrs)
    obs_points = dspn_utils.get_observation_points(data_al, log_model_align_map, net, im, semantic)
    training_sets, data_considered = get_training_sets(obs_points, categorical_attrs)
    data_considered = data_considered.difference(set(log['concept:name'].unique().tolist()))
    return  fit_classifiers(training_sets), obs_points, training_sets, data_considered
