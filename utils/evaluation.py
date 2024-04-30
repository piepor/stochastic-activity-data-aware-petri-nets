import copy
import pandas as pd
from tqdm import tqdm
import datetime
import time
import pm4py
from pm4py.objects.petri_net.semantics import ClassicSemantics
from utils.probabilities_utils import compute_traces_probability, add_dummies
from utils.general_utils import get_transition_by_name
from utils import alignments, probabilities_utils, general_utils, log_utils, conformance_measures
from utils.data_alignments import extract_variant_from_alignment
from conformance_modified import conformance_diagnostics_alignments


def evaluate_state(trace, net, im, fm, semantic, classifiers, categorical_attrs, not_data_attrs={'case:concept:name', 'time:timestamp', 'concept:name'}):
    # Evaluate state
    if trace['case:concept:name'].nunique() > 1:
        raise ValueError("Please evaluate the state on one trace at the time.")
    parameters = {alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    aligns = alignments.apply(trace, net, im, fm, parameters=parameters)
    # breakpoint()
    sequence = trace['concept:name'].tolist()
    # get to the state, possibly through multiple alignments
    probabilities = {'classifier': [], 'probability': [], 'transition': [], 'alignment': []}
    for count, align in enumerate(aligns):
        align_name = f"alignment-{count}"
        marking = im
        trace_attrs = trace.filter(regex='case:')
        data_sequence = [{attr: trace_attrs[attr].iloc[0] for attr in trace_attrs if not attr in not_data_attrs}]
        activity_count = {}
        count_sequence = 0
        data = copy.copy(trace_attrs.iloc[0])
        for name, label  in align['alignment']:
            trans_name = name[1]
            act_label = label[0]
            if act_label == sequence[count_sequence]:
                act = sequence[count_sequence]
                if not act in activity_count:
                    activity_count[act] = 0
                activity_count[act] += 1
                data = trace.iloc[count_sequence]
                count_sequence += 1
            if trans_name != '>>':
                # breakpoint()
                attrs = {attr: data[attr] for attr in data.index if not attr in not_data_attrs}
                data_sequence.append(copy.copy(attrs))
                trans = get_transition_by_name(net, trans_name)
                marking = semantic.execute(trans, net, marking) 
            if count_sequence == len(sequence):
                break
        enabled_trans = semantic.enabled_transitions(net, marking)
        data = {}
        for data_step in data_sequence:
            for attr in data_step:
                data[attr] = copy.copy(data_step[attr])
        for act in activity_count:
            data[act] = copy.copy(activity_count[act])
        data_df = pd.DataFrame(data, index=[0])
        for attr in data:
            probabilities[attr] = []
        for classifiers_type in classifiers:
            # breakpoint()
            trans_weights = {}
            total_weight = 0
            for trans in enabled_trans:
                if not classifiers_type == 'original':
                    features_order = classifiers[classifiers_type][trans.name].feature_names_in_
                    data_df = pd.DataFrame(data, index=[0])
                    data_df = add_dummies(data_df, categorical_attrs, features_order)
                    trans_weight = classifiers[classifiers_type][trans.name].predict_proba(data_df[features_order])[0, 1]
                    trans_weights[trans.name] = copy.copy(trans_weight)
                    total_weight += copy.copy(trans_weight)
                else:
                    trans_weight = 1
                    if trans.name == 'tau':
                        trans_weight = classifiers[classifiers_type][trans.name](data['B'])
                    elif trans.name == 'C':
                        trans_weight = classifiers[classifiers_type][trans.name](data['B'], data['case:Y'])
                    elif trans.name == 'D':
                        trans_weight = classifiers[classifiers_type][trans.name](data['B'], data['case:Y'])
                    trans_weights[trans.name] = copy.copy(trans_weight)
                    total_weight += copy.copy(trans_weight)
            for trans in enabled_trans:
                probabilities['alignment'].append(align_name)
                probabilities['classifier'].append(classifiers_type)
                probabilities['transition'].append(trans.name)
                probabilities['probability'].append(trans_weights[trans.name]/total_weight)
                for attr in data:
                    probabilities[attr].append(data[attr])
    # breakpoint()
    return pd.DataFrame.from_dict(probabilities)

def evaluate_states(traces, net, im, fm, semantic, classifiers, categorical_attrs, not_data_attrs={'case:concept:name', 'time:timestamp', 'concept:name'}):
    probabilities = pd.DataFrame()
    for case_id in traces['case:concept:name'].unique():
        trace = copy.copy(traces[traces['case:concept:name'] == case_id])
        probabilities_trace = copy.copy(evaluate_state(trace, net, im, fm, semantic, classifiers, categorical_attrs))
        probabilities = pd.concat([probabilities, probabilities_trace])
    return probabilities

def evaluate_example_states(net, im, fm, semantic, classifiers, categorical_attrs):
    curr_timestamp = time.time()
    traces = {'case:concept:name': ['0', '0', '1', '1', '1', '2', '2', '3', '3', '3'], 'concept:name': ['A', 'B', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'B'],
             'time:timestamp': [datetime.datetime.fromtimestamp(curr_timestamp),
                                datetime.datetime.fromtimestamp(curr_timestamp+1),
                                datetime.datetime.fromtimestamp(curr_timestamp),
                                datetime.datetime.fromtimestamp(curr_timestamp+1),
                                datetime.datetime.fromtimestamp(curr_timestamp+2),
                                datetime.datetime.fromtimestamp(curr_timestamp),
                                datetime.datetime.fromtimestamp(curr_timestamp+1),
                                datetime.datetime.fromtimestamp(curr_timestamp),
                                datetime.datetime.fromtimestamp(curr_timestamp+1),
                                datetime.datetime.fromtimestamp(curr_timestamp+2)],
                                         'case:Y': ['k', 'k', 'k', 'k', 'k', 'l', 'l', 'l', 'l', 'l']}
    traces = pd.DataFrame.from_dict(traces)
    return evaluate_states(traces, net, im, fm, semantic, classifiers, categorical_attrs)

def create_example_log():
    traces_lists = [['A', 'B', 'D'], ['A', 'B', 'D'], ['A', 'B', 'C'], ['A', 'B', 'C'],
                         ['A', 'B', 'B', 'D'], ['A', 'B', 'B', 'D'], ['A', 'B', 'B', 'C'], ['A', 'B', 'B', 'C'],
                         ['A', 'B', 'B', 'B', 'D'], ['A', 'B', 'B', 'B', 'D'], ['A', 'B', 'B', 'B', 'C'], ['A', 'B', 'B', 'B', 'C'],
                         ['A', 'B', 'B', 'B', 'B', 'D'], ['A', 'B', 'B', 'B', 'B',  'D'], ['A', 'B', 'B', 'B', 'B',  'C'], ['A', 'B', 'B', 'B', 'B',  'C'],
                    ]
    trace_attributes = [{'case:Y' : 'k'}, {'case:Y': 'l'}, {'case:Y': 'k'}, {'case:Y': 'l'},
                        {'case:Y' : 'k'}, {'case:Y': 'l'}, {'case:Y': 'k'}, {'case:Y': 'l'},
                        {'case:Y' : 'k'}, {'case:Y': 'l'}, {'case:Y': 'k'}, {'case:Y': 'l'},
                        {'case:Y' : 'k'}, {'case:Y': 'l'}, {'case:Y': 'k'}, {'case:Y': 'l'},
                    ]
    event_log = {'case:concept:name': [], 'concept:name': [], 'time:timestamp': [], 'case:Y': []}
    curr_timestamp = time.time()
    for i, trace in enumerate(traces_lists):
        for act in trace:
            event_log['case:concept:name'].append(str(i))
            event_log['concept:name'].append(act)
            event_log['case:Y'].append(trace_attributes[i]['case:Y'])
            event_log['time:timestamp'].append(datetime.datetime.fromtimestamp(curr_timestamp))
            curr_timestamp += 1
    return pd.DataFrame.from_dict(event_log)

def compute_example_trace_probabilities(classifiers, categorical_attrs, net, im, fm, semantic):
    example_log = create_example_log()
    df_prob = {'classifier': [], 'trace_probability': [], 'n_B': [], 'case:Y': [], 'last_act': [], 'case_id': []} 
    for classifiers_type in classifiers:
        print(f'{classifiers_type}')
        traces_probabilities = compute_traces_probability(
                example_log, classifiers[classifiers_type], categorical_attrs, net, im, fm, semantic, classifiers_type=classifiers_type)
        for case_id in tqdm(traces_probabilities['case_id'].unique()):
            prob = traces_probabilities[traces_probabilities['case_id'] == case_id]
            trace = example_log[example_log['case:concept:name'] == case_id]
            # breakpoint()
            act_seq = trace['concept:name'].tolist()
            n_b = act_seq.count('B')
            case_y = trace['case:Y'].unique()[0]
            last_act = act_seq[-1]
            df_prob['classifier'].append(classifiers_type)
            df_prob['trace_probability'].append(prob['model_probability'].iloc[0])
            df_prob['n_B'].append(n_b)
            df_prob['case:Y'].append(case_y)
            df_prob['last_act'].append(last_act)
            df_prob['case_id'].append(case_id)
    return pd.DataFrame.from_dict(df_prob)

def compute_duemsc(log, classifiers, net, im, fm, data_considered, categorical_attrs, not_data_attrs={'case:concept:name', 'time:timestamp', 'concept:name'}, semantic=ClassicSemantics()):
    data_sequences_log, count_data_sequences_log, count_data_evolutions, data_sequences_and_variants, map_data_sequences_log_model, map_data_sequences_log_to_data_evolution = log_utils.extract_log_data_sequences(log, data_considered)
    data_probabilities = []
    log_probabilities = []
    conditional_model_probabilities = {classifier: [] for classifier in classifiers}
    print("Computing probabilities...")
    # breakpoint()
    for data_sequence in tqdm(data_sequences_log):
        # data_probability = probabilities_utils.compute_data_probabilities(data_sequence, count_data_sequences_log)
        data_evolution = map_data_sequences_log_to_data_evolution[str(data_sequence)]
        data_probability = probabilities_utils.compute_data_probabilities(data_evolution, count_data_evolutions)
        for variant in data_sequences_and_variants[str(data_sequence)]:
            number_occurrences = data_sequences_and_variants[str(data_sequence)][variant]
            log_probability = probabilities_utils.compute_log_trace_probabilities(log, number_occurrences)
            data_sequence_model = map_data_sequences_log_model[str(data_sequence)][variant]
            trace = general_utils.assemble_trace(data_sequence_model, variant)
            trace = pm4py.convert_to_event_log(trace)
            # parameters = {alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
            # aligns = alignments.apply(trace, net, im, fm, parameters=parameters)
            aligns = conformance_diagnostics_alignments(trace, net, im, fm)
            trace = copy.copy(trace[0])
            # trace = general_utils.assemble_trace(data_sequence_model, variant)
            # trace = pm4py.convert_to_event_log(trace)[0]
            # parameters = {alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
            # aligns = alignments.apply(trace, net, im, fm, parameters=parameters)
            for classifiers_type in classifiers:
                trace_probability = 0
                for align in aligns:
                    # breakpoint()
                    if align['fitness'] == 1:
                        trace_probability += probabilities_utils.compute_trace_probability(
                                trace, align, classifiers[classifiers_type], net, im, semantic, categorical_attrs, not_data_attrs, classifiers_type)
                    conditional_model_probabilities[classifiers_type].append(trace_probability)
            log_probabilities.append(log_probability)
            data_probabilities.append(data_probability)
    duemscs = {'classifier': [], 'duemsc': []}
    for classifier in classifiers:
        moodel_probs = conditional_model_probabilities[classifier] 
        duemsc = conformance_measures.compute_duemsc(log_probabilities, moodel_probs, data_probabilities)
        duemscs['classifier'].append(classifier)
        duemscs['duemsc'].append(duemsc)
    return pd.DataFrame.from_dict(duemscs)

def compute_duemsc_for_analysis(log, classifiers, net, im, fm, data_considered, categorical_attrs, not_data_attrs={'case:concept:name', 'time:timestamp', 'concept:name'}, semantic=ClassicSemantics()):
    data_sequences_log, count_data_sequences_log, count_data_evolutions, data_sequences_and_variants, map_data_sequences_log_model, map_data_sequences_log_to_data_evolution = log_utils.extract_log_data_sequences(log, data_considered)
    data_probabilities = []
    log_probabilities = []
    avgs_fitness = []
    conditional_model_probabilities = {classifier: [] for classifier in classifiers}
    print("Computing probabilities...")
    # breakpoint()
    for data_sequence in tqdm(data_sequences_log):
        # data_probability = probabilities_utils.compute_data_probabilities(data_sequence, count_data_sequences_log)
        data_evolution = map_data_sequences_log_to_data_evolution[str(data_sequence)]
        data_probability = probabilities_utils.compute_data_probabilities(data_evolution, count_data_evolutions)
        for variant in data_sequences_and_variants[str(data_sequence)]:
            number_occurrences = data_sequences_and_variants[str(data_sequence)][variant]
            log_probability = probabilities_utils.compute_log_trace_probabilities(log, number_occurrences)
            data_sequence_model = map_data_sequences_log_model[str(data_sequence)][variant]
            trace = general_utils.assemble_trace(data_sequence_model, variant)
            trace = pm4py.convert_to_event_log(trace)
            # parameters = {alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
            # aligns = alignments.apply(trace, net, im, fm, parameters=parameters)
            aligns = conformance_diagnostics_alignments(trace, net, im, fm)
            trace = copy.copy(trace[0])
            # breakpoint()
            for classifiers_type in classifiers:
                trace_probability = 0
                for align in aligns:
                    avg_fitness = []
                    # breakpoint()
                    if align['fitness'] == 1:
                        trace_probability += probabilities_utils.compute_trace_probability(
                                trace, align, classifiers[classifiers_type], net, im, semantic, categorical_attrs, not_data_attrs, classifiers_type)
                        # breakpoint()
                    avg_fitness.append(copy.copy(align['fitness']))
                conditional_model_probabilities[classifiers_type].append(copy.copy(trace_probability))
                    # breakpoint()
            avgs_fitness.append(sum(copy.copy(avg_fitness))/len(copy.copy(avg_fitness)))
            log_probabilities.append(copy.copy(log_probability))
            data_probabilities.append(copy.copy(data_probability))
    # breakpoint()
    duemscs = {'classifier': [], 'duemsc': [], 'fitness': [], 'model_probability': [], 'log_probability': [], 'conditional_model_probabilities': [], 'data_probabilities': []}
    for classifier in classifiers:
        model_probs = conditional_model_probabilities[classifier] 
        traces_probabilities_model = copy.copy(list(map(lambda x, y: x*y, model_probs, data_probabilities)))
        num_traces = len(model_probs)
        duemsc = conformance_measures.compute_duemsc(log_probabilities, model_probs, data_probabilities)
        duemscs['classifier'].extend(copy.copy(num_traces*[classifier]))
        duemscs['duemsc'].extend(copy.copy(num_traces*[duemsc]))
        duemscs['fitness'].extend(copy.copy(avgs_fitness))
        duemscs['model_probability'].extend(copy.copy(traces_probabilities_model))
        duemscs['log_probability'].extend(copy.copy(log_probabilities))
        duemscs['conditional_model_probabilities'].extend(copy.copy(model_probs))
        duemscs['data_probabilities'].extend(copy.copy(data_probabilities))
    return pd.DataFrame.from_dict(duemscs)
