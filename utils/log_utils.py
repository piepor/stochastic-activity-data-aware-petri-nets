import os
import pm4py
import copy
from tqdm import tqdm
from utils import simulation


def import_log(log_name: str, simulate=False):
    if f'{log_name}.xes' not in os.listdir('logs') or simulate:
        # DA FARE
        if log_name in ['data-aware-spn-paper', 'sequence-data-aware-spn']:
            log = simulation.simulate(log_name, no_traces=10000)
        else:
            raise NotImplementedError('Model not available')
        pm4py.write_xes(log, f'./logs/{log_name}.xes')
        log_legacy = copy.deepcopy(log)
        log = pm4py.convert_to_dataframe(log)
    else:
        log = pm4py.read_xes(f'./logs/{log_name}.xes')
        log_legacy = pm4py.read_xes(f'./logs/{log_name}.xes', return_legacy_log_object=True)
    return log, log_legacy

def extract_log_data_sequences(log, data_considered):
    data_sequences_log = []
    count_data_evolutions = {}
    count_data_sequences_log = {}
    map_data_sequences_log_model = {}
    map_data_sequences_log_to_data_evolution = {}
    data_sequences_and_variants = {}
    print("Extracting log data sequences...")
    for case_id in tqdm(log['case:concept:name'].unique()):
    # for case_id in log['case:concept:name'].unique():
        trace = log[log['case:concept:name'] == case_id]
        trace_attrs = trace.filter(regex='case:')
        # breakpoint()
        trace_log = trace[trace.columns[~trace.columns.isin(trace_attrs)]]
        data_state = {attr: trace_attrs[attr].iloc[0] for attr in trace_attrs if attr in data_considered}
        data_sequence = [copy.copy(data_state)]
        data_state_log = {attr: trace_attrs[attr].iloc[0] for attr in trace_attrs if attr in data_considered}
        data_sequence_log = [copy.copy(data_state_log)]
        data_evolution = {attr: trace_attrs[attr].iloc[0] for attr in trace_attrs if attr in data_considered}
        variant = tuple(trace['concept:name'].tolist())
        for row, data in trace.iterrows():
            # breakpoint()
            attrs = {attr: data[attr] for attr in data.index if attr in data_considered}
            for attr in attrs:
                if not attr in trace_attrs:
                    if attr in data_evolution:
                        data_evolution[attr].append(attrs[attr])
                    else:
                        data_evolution[attr] = [attrs[attr]]
            # try:
            data_state.update(attrs)
            data_sequence.append(copy.copy(data_state))
            attrs_log = {attr: trace_log.loc[row][attr] for attr in trace_log.loc[row].index if attr in data_considered}
            data_state_log.update(attrs_log)
            # except:
            if attrs_log:
                data_sequence_log.append(copy.copy(data_state_log))
        if not data_sequence_log in data_sequences_log:
            # breakpoint()
            data_sequences_log.append(copy.copy(data_sequence_log))
            count_data_sequences_log[str(data_sequence_log)] = 1
            map_data_sequences_log_model[str(data_sequence_log)] = {variant: data_sequence}
            map_data_sequences_log_to_data_evolution[str(data_sequence_log)] =  copy.copy(data_evolution)
            data_sequences_and_variants[str(data_sequence_log)] = {variant: 1}
        else:
            # breakpoint()
            count_data_sequences_log[str(data_sequence_log)] += 1
            # breakpoint()
            if variant in data_sequences_and_variants[str(data_sequence_log)]:
                data_sequences_and_variants[str(data_sequence_log)][variant] += 1
            else:
                data_sequences_and_variants[str(data_sequence_log)][variant] = 1
                map_data_sequences_log_model[str(data_sequence_log)][variant] =  copy.copy(data_sequence)
        if not str(data_evolution) in count_data_evolutions:
            count_data_evolutions[str(data_evolution)] = 1
        else:
            count_data_evolutions[str(data_evolution)] += 1
        # breakpoint()
        # data_sequence_and_variant = tuple([tuple(variant), data_sequence, data_sequence_log])
        # if not data_sequence_and_variant in data_sequences_and_variants:
        #     data_sequences_and_variants.append(data_sequence_and_variant)
    # breakpoint()
    return data_sequences_log, count_data_sequences_log, count_data_evolutions, data_sequences_and_variants, map_data_sequences_log_model, map_data_sequences_log_to_data_evolution

# def extract_log_data_sequences(log, not_data_attrs={'case:concept:name', 'time:timestamp', 'concept:name'}):
#     data_sequences_log = []
#     count_data_sequences_log = {}
#     data_sequences_and_variants = []
#     # for case_id in tqdm(log['case:concept:name'].unique()):
#     for case_id in log['case:concept:name'].unique():
#         trace = log[log['case:concept:name'] == case_id]
#         trace_attrs = trace.filter(regex='case:')
#         # breakpoint()
#         trace_log = trace[trace.columns[~trace.columns.isin(trace_attrs)]]
#         data_sequence = [{attr: trace_attrs[attr].iloc[0] for attr in trace_attrs if not attr in not_data_attrs}]
#         data_sequence_log = [{attr: trace_attrs[attr].iloc[0] for attr in trace_attrs if not attr in not_data_attrs}]
#         variant = trace['concept:name'].tolist()
#         for row, data in trace.iterrows():
#             # breakpoint()
#             attrs = {attr: data[attr] for attr in data.index if not attr in not_data_attrs}
#             # try:
#             attrs_log = {attr: trace_log.loc[row][attr] for attr in trace_log.loc[row].index if not attr in not_data_attrs}
#             # except:
#             data_sequence.append(copy.copy(attrs))
#             if attrs_log:
#                 data_sequence_log.append(copy.copy(attrs_log))
#         if not data_sequence_log in data_sequences_log:
#             data_sequences_log.append(data_sequence_log)
#             count_data_sequences_log[str(data_sequence_log)] = 1
#         else:
#             count_data_sequences_log[str(data_sequence_log)] += 1
#         breakpoint()
#         data_sequence_and_variant = tuple([tuple(variant), data_sequence, data_sequence_log])
#         if not data_sequence_and_variant in data_sequences_and_variants:
#             data_sequences_and_variants.append(data_sequence_and_variant)
#     return data_sequences_log, count_data_sequences_log, data_sequences_and_variants
