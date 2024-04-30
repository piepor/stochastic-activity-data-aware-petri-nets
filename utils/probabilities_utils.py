import types
from pm4py.objects.petri_net.semantics import ClassicSemantics
import pm4py
import pandas as pd
import copy
import math
from tqdm import tqdm
import utils.dijkstra_local as generator_dijkstra_less_memory
import utils.activities_data_aware_spn_utils as adspn_utils
from utils.data_alignments import get_data_sequence, extract_variant_from_alignment, extract_model_path_from_alignment
from utils import alignments


def add_dummies(df, categorical_attrs, features_order):
    # get dummies for the categorical attributes present
    data_df = pd.get_dummies(df, columns=categorical_attrs)
    # check if some traininig features of the classifier is not present 
    # for example the other dummies values different from the one present in this trace
    not_present_attrs = set(features_order).difference(set(data_df.columns))
    for attr in not_present_attrs:
        # if the attribute not present is categorical the default value is False, 
        # if continuous 0
        # WARNING "_" is used to determine the dummy part of the name: DO NOT USE ATTRIBUTES WITH "_"
        if attr.split("_")[0] in categorical_attrs:
            data_df[attr] = False
        else:
            data_df[attr] = 0
    return data_df

def compute_trans_probability(enabled_trans, data, activity_counter, fired_trans_name, categorical_attrs, classifiers, classifiers_type, not_data_attrs):
    # breakpoint()
    if len(enabled_trans) == 1:
        prob = 1
    else:
        attrs_and_acts = copy.copy(data)
        attrs_and_acts.update(activity_counter)
        total_weight = 0
        weight = -1
        prob = -1
        if not classifiers_type == 'original':
            for trans in enabled_trans:
                if not isinstance(classifiers[trans.name], types.FunctionType):
                    features_order = classifiers[trans.name].feature_names_in_
                    data_df = pd.DataFrame(attrs_and_acts, index=[0])
                    categorical_attrs_present = list(set(categorical_attrs).intersection(set(data_df.columns)))
                    # try:
                    data_df = add_dummies(data_df, categorical_attrs_present, features_order)
                    # except:
                    #     breakpoint()
                    if trans.name == fired_trans_name:
                        weight = copy.copy(classifiers[trans.name].predict_proba(data_df[features_order])[0, 1])
                    total_weight += copy.copy(classifiers[trans.name].predict_proba(data_df[features_order])[0, 1])
                else:
                    if trans.name == fired_trans_name:
                        weight = copy.copy(classifiers[trans.name]())
                    total_weight += copy.copy(classifiers[trans.name]())
            if weight == -1:
                raise ValueError("Weight not assigned to any transition.")
            prob = weight/total_weight
        else:
            prob = 1
            if fired_trans_name == 'tau':
                prob = classifiers[fired_trans_name](attrs_and_acts['B'])
            elif fired_trans_name == 'C':
                prob = classifiers[fired_trans_name](attrs_and_acts['B'], attrs_and_acts['case:Y'])
            elif fired_trans_name == 'D':
                prob = classifiers[fired_trans_name](attrs_and_acts['B'], attrs_and_acts['case:Y'])
        if prob == -1:
            raise ValueError("Prob not assigned to any transition.")
    return prob

def compute_state_trans_probabilities(enabled_trans, data, classifiers, classifiers_type, categorical_attrs):
    total_weight = 0
    trans_weights = {}
    for trans in enabled_trans:
        if not classifiers_type == 'original':
            features_order = classifiers[trans.name].feature_names_in_
            data_df = pd.DataFrame(data, index=[0])
            data_df = add_dummies(data_df, categorical_attrs, features_order)
            trans_weight = classifiers[trans.name].predict_proba(data_df[features_order])[0, 1]
            trans_weights[trans.name] = copy.copy(trans_weight)
            total_weight += copy.copy(trans_weight)
        else:
            trans_weight = 1
            if trans.name == 'tau':
                trans_weight = classifiers[trans.name](data['B'])
            elif trans.name == 'C':
                trans_weight = classifiers[trans.name](data['B'], data['case:Y'])
            elif trans.name == 'D':
                trans_weight = classifiers[trans.name](data['B'], data['case:Y'])
            trans_weights[trans.name] = copy.copy(trans_weight)
            total_weight += copy.copy(trans_weight)
    return {trans_name: trans_weights[trans_name]/total_weight for trans_name in trans_weights}

def compute_trace_probability(trace, align, classifiers, net, im, semantic, categorical_attrs, not_data_attrs, classifiers_type):
    marking = im
    # breakpoint()
    if align['fitness'] == 1:
        # breakpoint()
        data_sequence = get_data_sequence(trace, align, not_data_attrs)
        model_path = extract_model_path_from_alignment(align)
        probs = []
        activity_counter = {}
        for counter, fired_trans_name in enumerate(model_path):
            enabled_trans = semantic.enabled_transitions(net, marking)
            prob = compute_trans_probability(
                    enabled_trans, data_sequence[counter], activity_counter, fired_trans_name, categorical_attrs, classifiers, classifiers_type, not_data_attrs)
            probs.append(copy.copy(prob))
            trans_to_fire = adspn_utils.get_transition_by_name(net, fired_trans_name)
            if trans_to_fire.label:
                if not trans_to_fire.label in activity_counter:
                    activity_counter[trans_to_fire.label] = 1
                else:
                    activity_counter[trans_to_fire.label] += 1
            marking = semantic.execute(trans_to_fire, net, marking) 
    else:
        probs = [0]
    return math.prod(probs)

def compute_traces_probability(
        traces, classifiers, categorical_attrs, net, im, fm, semantic, 
        not_data_attrs={'concept:name', 'time:timestamp', 'case:concept:name'}, classifiers_type='activity-data-aware'):
    parameters = {alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    aligns = alignments.apply(traces, net, im, fm, parameters=parameters)
    traces = pm4py.convert_to_event_log(traces)
    traces_probabilities = {'case_id': [], 'model_probability': []}
    for trace in tqdm(traces):
        trace_probability = 0
        variant = tuple([event['concept:name'] for event in trace])
        for align in aligns:
            if extract_variant_from_alignment(align) == variant:
                trace_probability += compute_trace_probability(
                        trace, align, classifiers, net, im, semantic, categorical_attrs, not_data_attrs, classifiers_type)
        traces_probabilities['case_id'].append(trace.attributes['concept:name'])
        traces_probabilities['model_probability'].append(trace_probability)
    return pd.DataFrame.from_dict(traces_probabilities)

def compute_log_trace_probabilities(log, number_occurrences):
    # count_data_and_activities_seq = 0
    # # for case_id in tqdm(log['case:concept:name'].unique()):
    # for case_id in log['case:concept:name'].unique():
    #     trace = log[log['case:concept:name'] == case_id]
    #     trace_attrs = trace.filter(regex='case:')
    #     data_sequence = [{attr: trace_attrs[attr].iloc[0] for attr in trace_attrs if not attr in not_data_attrs}]
    #     trace = trace[trace.columns[~trace.columns.isin(trace_attrs)]]
    #     for _, data in trace.iterrows():
    #         attrs = {attr: data[attr] for attr in data.index if not attr in not_data_attrs}
    #         if attrs:
    #             data_sequence.append(copy.copy(attrs))
    #     variant = tuple(trace['concept:name'].tolist())
    #     if  data_sequence == data_sequence_in and variant == variant_in:
    #         count_data_and_activities_seq += 1
    return number_occurrences / log['case:concept:name'].nunique()

# def compute_log_trace_probabilities(log, data_sequence_in, variant_in, not_data_attrs={'concept:name', 'time:timestamp', 'case:concept:name'}):
#     count_data_and_activities_seq = 0
#     # for case_id in tqdm(log['case:concept:name'].unique()):
#     for case_id in log['case:concept:name'].unique():
#         trace = log[log['case:concept:name'] == case_id]
#         trace_attrs = trace.filter(regex='case:')
#         data_sequence = [{attr: trace_attrs[attr].iloc[0] for attr in trace_attrs if not attr in not_data_attrs}]
#         trace = trace[trace.columns[~trace.columns.isin(trace_attrs)]]
#         for _, data in trace.iterrows():
#             attrs = {attr: data[attr] for attr in data.index if not attr in not_data_attrs}
#             if attrs:
#                 data_sequence.append(copy.copy(attrs))
#         variant = tuple(trace['concept:name'].tolist())
#         if  data_sequence == data_sequence_in and variant == variant_in:
#             count_data_and_activities_seq += 1
#     return count_data_and_activities_seq / log['case:concept:name'].nunique()

def compute_log_trace_probabilities_old(trace_in, log, not_data_attrs):
    num_equal = 0
    for trace in log:
        acts_in = copy.copy([event['concept:name'] for event in trace_in])
        acts = copy.copy([event['concept:name'] for event in trace])
        n_eq_trace_attrs = 0
        eq_trace_attrs = False
        for trace_attr in set(trace.attributes).difference(set(not_data_attrs)):
            if trace.attributes[trace_attr] == trace_in.attributes[f'case:{trace_attr}']:
                n_eq_trace_attrs += 1
        if n_eq_trace_attrs == len(set(trace.attributes).difference(set(not_data_attrs))):
            eq_trace_attrs = True
        if acts_in == acts and eq_trace_attrs:
            num_equal += 1
    return num_equal/len(log)

def compute_data_probabilities(data_sequence, count_data_sequences):
    total_ds = sum([count_data_sequences[str(ds)] for ds in count_data_sequences])
    return count_data_sequences[str(data_sequence)] / total_ds

def compute_data_probabilities_old(data, log, not_data_attrs):
    num_equal = 0
    for trace in log:
        n_eq_trace_attrs = 0
        eq_trace_attrs = False
        for trace_attr in set(trace.attributes).difference(set(not_data_attrs)):
            if trace.attributes[trace_attr] == data[f'case:{trace_attr}']:
                n_eq_trace_attrs += 1
        if n_eq_trace_attrs == len(set(trace.attributes).difference(set(not_data_attrs))):
            num_equal += 1
    return num_equal/len(log)
