import pandas as pd
from pm4py.objects.log.obj import Trace, Event
import copy
import time
import datetime


def replace_underscore_in_columns(df):
    return df.rename(columns=lambda c: c.replace('_', '-'))

def get_cat_columns_maximum_values_onehot(df, columns, max_one_hot):
    return [column for column in columns if df[column].nunique() <= max_one_hot]

def get_transition_by_name(net, trans_name):
    return [trans for trans in net.transitions if trans.name == trans_name][0]

def get_activities_perc(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(len(df.columns), 'total_num', 1)
    ptable = df.pivot_table(index=['n_b', 'act'], values='total_num', aggfunc='sum') 
    perc_table = ptable['total_num'] / ptable.groupby(level=0)['total_num'].sum()
    return perc_table.reset_index().rename(columns={'total_num': 'perc'})

def get_data_perc(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(len(df.columns), 'total_num', 1)
    ptable = df.pivot_table(index=['case:Y', 'concept:name'], values='total_num', aggfunc='sum') 
    perc_table = ptable['total_num'] / ptable.groupby(level=0)['total_num'].sum()
    return perc_table.reset_index().rename(columns={'total_num': 'perc'})

def convert_sequence_from_string(sequence_str):
    return sequence_str.replace(",", "").replace("'", "").replace("[", "").replace("]", "").replace(" ", "")

def count_number(df):
    # breakpoint()
    number_df = {'n_b': [], 'posix': [], 'act': []}
    variants_df = df[['case:concept:name', 'concept:name']].groupby('case:concept:name').agg(lambda x: str(list(x)))
    variants_df['concept:name'] = variants_df['concept:name'].apply(convert_sequence_from_string)
    for _, row in variants_df.iterrows():
        prefix = []
        for i, act in enumerate(row['concept:name']):
            n_b = prefix.count('B')
            number_df['n_b'].append(n_b)
            number_df['posix'].append(i)
            number_df['act'].append(act)
            prefix.append(act)
    # breakpoint()
    return pd.DataFrame.from_dict(number_df)

def assemble_trace(data_sequence, variant):
    df = {'case:concept:name': [], 'concept:name': [], 'time:timestamp': []}
    curr_timestamp = time.time()
    for counter, act in enumerate(variant):
        df['case:concept:name'].append('0')
        df['concept:name'].append(act)
        df['time:timestamp'].append(datetime.datetime.fromtimestamp(curr_timestamp))
        for attr in data_sequence[counter+1]:
            if attr not in df:
                df[attr] = []
            df[attr].append(data_sequence[counter+1][attr])
    return pd.DataFrame.from_dict(df)

def assemble_trace_old(trace_activities, trace_attributes):
    df = {'case:concept:name': len(trace_activities)*['0'], 'concept:name': [act for act in trace_activities]}
    trace = Trace()
    for trace_attr in trace_attributes:
        trace.attributes['case:concept:name'] = '0'
        trace.attributes[trace_attr] = trace_attributes[trace_attr]
        df[trace_attr] = len(trace_activities)*[trace_attributes[trace_attr]]
    for act in trace_activities:
        event = Event()
        event['concept:name'] = act
        trace.append(copy.copy(event))
    return copy.copy(trace), pd.DataFrame.from_dict(df)
