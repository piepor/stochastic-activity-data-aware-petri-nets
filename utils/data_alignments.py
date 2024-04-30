import pm4py
import copy
import pandas as pd
from utils import alignments
from tqdm import tqdm
from conformance_modified import conformance_diagnostics_alignments


def extract_alignment_from_model_path(alignments, model_path):
    return [align for align in alignments if extract_model_path_from_alignment(align) == model_path][0]

def extract_log_alignment_from_model_path(alignments, model_path):
    return [extract_log_from_alignment(align) for align in alignments if extract_model_path_from_alignment(align) == model_path][0]

def extract_variant_from_alignment(alignment):
    return tuple([trans[1][0] for trans in alignment['alignment'] if trans[1][0] != '>>'])

def extract_model_path_from_alignment(alignment):
    return tuple([trans[0][1] for trans in alignment['alignment']])

def extract_log_from_alignment(alignment):
    return tuple([trans[0][0] for trans in alignment['alignment']])

def get_data_sequence(trace, align, not_data_attrs):
    data_state = {f"case:{attr}": trace.attributes[attr] for attr in trace.attributes if not attr in not_data_attrs}
    data_sequence = [copy.copy(data_state)]
    cont_trace = 0
    for names, labels in align['alignment']:
        log_activity = labels[0]
        model_label = labels[1]
        if not model_label:
            event = copy.copy(trace[cont_trace-1])
        else:
            event = copy.copy(trace[cont_trace])
        # breakpoint()
        # try:
        attrs = {attr: event[attr] for attr in event if not attr in not_data_attrs}
        # except:
        #     breakpoint()
        #attrs.update(data_sequence[0])
        data_state.update(attrs)
        data_sequence.append(copy.copy(data_state))
        if model_label == '>>' and log_activity != '>>':
            print(align['alignment'])
            print(labels)
            print(names)
            raise ValueError('Assumption no log moves violated')
        # increase counter only if it is not already the end of the trace (final activity could be in the middle
        # of the alignment). If later in the alignment there is another log activity different from '>>',
        # the above condition will not hold
        elif log_activity != '>>' and cont_trace < len(trace)-1:
            cont_trace += 1
    return copy.copy(data_sequence)

def check_for_log_moves(align):
    log_move_found = False
    for _, labels in align['alignment']:
        # breakpoint()
        log_activity = labels[0]
        model_label = labels[1]
        if model_label == '>>' and log_activity != '>>':
            log_move_found = True
            break
    return log_move_found

def extract(log, net, im, fm, not_data_attrs={'concept:name', 'time:timestamp', 'case:concept:name'}):
    data_sequences = {}
    variants = pm4py.get_variants_as_tuples(pm4py.convert_to_event_log(log))
    # parameters = {alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    # aligns = alignments.apply(log, net, im, fm, parameters=parameters)
    aligns = conformance_diagnostics_alignments(log, net, im, fm)
    # aligns_3 = pm4py.conformance_diagnostics_alignments(log, net, im, fm)
    # breakpoint()
    model_log_alignment_map = {}
    for align in tqdm(aligns):
        check_log_move = check_for_log_moves(align)
        # use only alignments without log moves (log activities not included in the model)
        if not check_log_move:
            # breakpoint()
            model_path = extract_model_path_from_alignment(align)
            data_sequences[model_path] = []
            align_variant = extract_variant_from_alignment(align)
            variant_traces = variants[align_variant]
            for trace in variant_traces:
                data_sequences[model_path].append(get_data_sequence(trace, align, not_data_attrs)) 
            if not model_path in model_log_alignment_map:
                model_log_alignment_map[model_path] = extract_log_alignment_from_model_path(aligns, model_path)
    return data_sequences, model_log_alignment_map

def group_data_step(data_alignment):
    # breakpoint()
    overall_data_step = {f"step-{counter}": {} for counter in range(len(data_alignment[0]))}
    for data in data_alignment:
        for counter, data_step in enumerate(data):
            step_name = f"step-{counter}"
            for attr in data_step:
                if attr not in overall_data_step[step_name]:
                    if len(overall_data_step[step_name]) == 0:
                        # add first column
                        overall_data_step[step_name][attr] = [data_step[attr]]
                    else:
                        # if "length" in the keys, it means that it is not the first time filling the dict:
                        # the number of "length" is the right one since it is updated last and outside the loop.
                        if "length" in overall_data_step[step_name]:
                            dict_length = len(overall_data_step[step_name]["length"])
                            overall_data_step[step_name][attr] = dict_length*[None]
                        else:
                            # if not "length" in the keys, it means that it is the first time filling the dictionaries and we 
                            # do not have to add None values.
                            overall_data_step[step_name][attr] = []
                        overall_data_step[step_name][attr].append(data_step[attr])
                else:
                    overall_data_step[step_name][attr].append(data_step[attr])
            # add the column 'length' for construnction purposes
            if 'length' not in overall_data_step[step_name]:
                overall_data_step[step_name]['length'] = []
            # add None to all attributes not present in data_step
            overall_data_step[step_name]['length'].append(None)
            attr_not_present = set(overall_data_step[step_name].keys()).difference(set(data_step.keys()))
            for attr in attr_not_present:
                if attr != 'length':
                    overall_data_step[step_name][attr].append(None)
    # return the dictionary without the column used to track data insertion
    # breakpoint()
    for step_name in overall_data_step:
        overall_data_step[step_name].pop('length')
    # breakpoint()
    return overall_data_step

# def order_alignments(log, alignments):
#     ordered_alignments = {}
#     for case_id in log['case:concept:name'].unique():
#         ordered_alignments[case_id] = []
#         variant = log[log['case:concept:name'] == case_id]['concept:name'].tolist()
#         for align in alignments:
#             variant_align = extract_variant_from_alignment(align)
#             if variant == variant_align:
#                 # modified to accept multiple alignments
#                 ordered_alignments[case_id].append(copy.copy(align))
#     return ordered_alignments
