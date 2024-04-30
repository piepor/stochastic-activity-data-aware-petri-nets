import pm4py
import copy
import pandas as pd
from utils import alignments
from tqdm import tqdm


def extract_alignment_from_model_path(alignments, model_path):
    return [align for align in alignments if extract_model_path_from_alignment(align) == model_path]

def extract_variant_from_alignment(alignment):
    return tuple([trans[1][0] for trans in alignment['alignment'] if trans[1][0] != '>>'])

def extract_model_path_from_alignment(alignment):
    return tuple([trans[0][1] for trans in alignment['alignment']])

def get_data_sequence(trace, align, not_data_attrs):
    data_sequence = [{f"case:{attr}": trace.attributes[attr] for attr in trace.attributes if not attr in not_data_attrs}]
    cont_trace = 0
    for names, labels in align['alignment']:
        log_activity = labels[0]
        model_label = labels[1]
        event = trace[cont_trace]
        attrs = {attr: event[attr] for attr in event if not attr in not_data_attrs}
        # add trace attributes
        # breakpoint()
        attrs.update(data_sequence[0])
        data_sequence.append(attrs)
        if log_activity != '>>':
            cont_trace += 1
        elif model_label == '>>' and log_activity != '>>':
            print(align['alignment'])
            print(labels)
            print(names)
            raise ValueError('Assumption no log moves violated')
    return data_sequence

def extract(log, net, im, fm, not_data_attrs={'concept:name', 'time:timestamp', 'case:concept:name'}):
    data_sequences = {}
    variants = pm4py.get_variants_as_tuples(pm4py.convert_to_event_log(log))
    parameters = {alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    aligns = alignments.apply(log, net, im, fm, parameters=parameters)
    for align in tqdm(aligns):
        # breakpoint()
        model_path = extract_model_path_from_alignment(align)
        data_sequences[model_path] = []
        align_variant = extract_variant_from_alignment(align)
        variant_traces = variants[align_variant]
        for trace in variant_traces:
            data_sequences[model_path].append(get_data_sequence(trace, align, not_data_attrs)) 
    return data_sequences

def order_alignments(log, alignments):
    ordered_alignments = {}
    for case_id in log['case:concept:name'].unique():
        ordered_alignments[case_id] = []
        variant = log[log['case:concept:name'] == case_id]['concept:name'].tolist()
        for align in alignments:
            variant_align = extract_variant_from_alignment(align)
            if variant == variant_align:
                # modified to accept multiple alignments
                ordered_alignments[case_id].append(copy.copy(align))
    return ordered_alignments

# def get_alignment_with_data(align_sequence, data_sequence, trace_attrs):
#     trace_attrs = copy.copy(trace_attrs)
#     data_alignment = {'data': [], 'log': [], 'model': [], 'trace': trace_attrs}
#     cont_data_sequence = 0
#     for names, labels in align_sequence:
#         trans_name = names[1]
#         trans_label = labels[1]
#         data_alignment['model'].append(trans_name)
#         log_label_move = labels[0]
#         data_alignment['log'].append(log_label_move)
#         if cont_data_sequence == len(data_sequence):
#             cont_data_sequence -= 1
#         data_alignment['data'].append(data_sequence.iloc[cont_data_sequence])
#         if log_label_move != '>>' and not trans_label in [None, '>>']: 
#             cont_data_sequence += 1
#         elif trans_label == '>>' and log_label_move != '>>':
#             print(align_sequence)
#             print(labels)
#             print(names)
#             raise ValueError('Assumption no log moves violated')
#     return data_alignment

# def alignments_with_data(alignments: list, data: pd.DataFrame,
#                          not_data_attrs: set = {'case:concept:name', 'concept:name', 'time:timestamp'}):
#     data_alignments = []
#     traces_attrs = (data
#                    .filter(regex='case:')
#                    .drop_duplicates('case:concept:name')
#                    .reset_index(drop=True))
#     for i, case_id in enumerate(alignments):
#         trace_attrs = traces_attrs.iloc[i]
#         trace_attrs_names = [col for col in traces_attrs.columns if not col in not_data_attrs]
#         case_id = trace_attrs['case:concept:name']
#         data_columns = [col for col in data.columns if not col in not_data_attrs]
#         data_sequence = data[data['case:concept:name'] == case_id][data_columns]
#         # data_alignment = {'data': [], 'log': [], 'model': [], 'trace': trace_attr[trace_attr_names]}
#         for alignment in alignments[case_id]:
#             align_sequence = alignment['alignment']
#             data_alignment = get_alignment_with_data(align_sequence, data_sequence, trace_attrs[trace_attrs_names])
#             data_alignments.append(data_alignment)
#     return data_alignments
