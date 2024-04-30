import utils.general_utils as utils
import plotly.express as px
import pandas as pd
import pm4py
import copy
import numpy as np


def plot_analysis_results(results):
    fig = px.scatter(results, x='conditional_model_probabilities', y='fitness', color='classifier', facet_row='log', facet_col='noise-threshold')
    fig.write_html('./figures/html/conditional-model-probabilities-fitness.html')
    fig.write_image('./figures/svg/conditional-model-probabilities-fitness.svg')
    fig.show()
    fig = px.scatter(results, x='model_probability', y='log_probability', color='classifier', facet_row='log', facet_col='noise-threshold')
    fig.write_html('./figures/html/log-model-probability.html')
    fig.write_image('./figures/svg/log-model-probability.svg')
    fig.show()
    fig = px.scatter(results, x='conditional_model_probabilities', y='log_probability', color='classifier', facet_row='log', facet_col='noise-threshold')
    fig.write_html('./figures/html/log-conditional-model-probability.html')
    fig.write_image('./figures/svg/log-conditional-model-probability.svg')
    fig.show()
    fig = px.scatter(results, x='conditional_model_probabilities', y='data_probabilities', color='classifier', facet_row='log', facet_col='noise-threshold')
    fig.write_html('./figures/html/data-conditional-model-probability.html')
    fig.write_image('./figures/svg/data-conditional-model-probability.svg')
    fig.show()
    # breakpoint()
    results['model'] = results.apply(lambda row: f"{row['classifier']}-{row['noise-threshold']}", axis=1)
    results =  results.drop_duplicates('model')
    fig = px.bar(results, x='classifier', y='duemsc', color='noise-threshold', barmode='group')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html('./figures/html/duemsc-model-type.html')
    fig.write_image('./figures/svg/duemsc-model-type.svg')
    fig.show()

def plot_results(results):
    fig = px.box(results, x='log', y='duemsc', color='model-type', facet_row='noise-threshold')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html('./figures/html/results.html')
    fig.write_image('./figures/svg/results.svg')
    fig.show()

def plot_log_activities_stats(log):
    stats_df_k = utils.count_number(log[log['case:Y'] == 'k'])
    stats_df_l = utils.count_number(log[log['case:Y'] == 'l'])
    # breakpoint()
    perc_k = utils.get_activities_perc(stats_df_k[stats_df_k['act'] != 'A'])
    perc_l = utils.get_activities_perc(stats_df_l[stats_df_l['act'] != 'A'])
    perc_k.insert(len(perc_k.columns), 'case:Y', 'k')
    perc_l.insert(len(perc_l.columns), 'case:Y', 'l')
    # breakpoint()
    perc = pd.concat([perc_k, perc_l])
    # breakpoint()
    fig = px.bar(perc, x='n_b', y='perc', color='act', barmode='group', facet_col='case:Y')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html('./figures/html/log_acts_perc.html')
    fig.write_image('./figures/svg/log_acts_perc.svg')
    fig.show()
    log_case = log[log['concept:name'].isin(['C', 'D'])].drop_duplicates(subset=['case:concept:name'])
    fig = px.histogram(log_case, 'case:Y', color='concept:name', barmode='group')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html('./figures/html/log_CD_hist.html')
    fig.write_image('./figures/svg/log_CD_hist.svg')
    fig.show()
    prova = utils.get_data_perc(log_case)
    fig = px.bar(prova, x='case:Y', y='perc', color='concept:name', barmode='group')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html('./figures/html/log_CD_perc.html')
    fig.write_image('./figures/svg/log_CD_perc.svg')
    fig.show()

def plot_duemsc(df):
    fig = px.bar(df, x='classifier', y='duemsc')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html('./figures/html/duemsc.html')
    fig.write_image('./figures/svg/duemsc.svg')
    fig.show()

def plot_trace_probability(df):
    fig = px.bar(df, x='n_B', y='trace_probability', color='classifier', barmode='group', facet_col='case:Y', facet_row='last_act')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html('./figures/html/trace_probability.html')
    fig.write_image('./figures/svg/trace_probability.svg')
    fig.show()

def plot_activities_probability(df):
    fig = px.bar(df, x='transition', y='probability', color='classifier', barmode='group', facet_col='case:Y', facet_row='B')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html('./figures/html/activities_probability.html')
    fig.write_image('./figures/svg/activities_probability.svg')
    fig.show()

def plot_petri_net(net, im, fm):
    pm4py.vis.save_vis_petri_net(net, im, fm, './figures/svg/sequence_data_aware_model.svg')

def plot_act_perc(df, act):
    fig = px.histogram(df, 'case:Y', color='label', barmode='group')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html(f'./figures/html/activities_percentage{act}.html')
    fig.write_image(f'./figures/svg/activities_percentage{act}.svg')
    fig.show()

def plot_acts_perc(labels, observation_points_activities_and_data, label_transition_map, transition_label_map):
    for label in labels:
        obs = copy.copy(observation_points_activities_and_data[label_transition_map[label]])
        obs['label'] = [transition_label_map[trans] for trans in obs['fired']]
        df = pd.DataFrame.from_dict(obs)
        plot_act_perc(df, label)

def plot_weight_functions_sequence_version(classifiers, transitions, training_sets):
    results = {'A': [], 'B': [], 'case:Y_k': [], 'case:Y_l': [],'weight': [], 'label': [], 'case:Y': []} 
    case_ys = ['k', 'l']
    for case_y in case_ys:
        data = {}
        data['A'] = 10*[1]
        data['B'] = np.arange(10)
        if case_y == 'l':
            data['case:Y_k'] = 10*[False]
            data['case:Y_l'] = 10*[True]
        else:
            data['case:Y_l'] = 10*[False]
            data['case:Y_k'] = 10*[True]
        data = pd.DataFrame.from_dict(data)
        for trans_name, trans_label in transitions:
            features_trained = training_sets[trans_name].columns.drop('target').tolist()
            for feature in features_trained:
                results[feature].extend(data[feature].tolist())
            if not trans_label:
                trans_label = trans_name
            results['label'].extend(len(data['case:Y_k'])*[trans_label])
            classifier = classifiers[trans_name]
            results['weight'].extend(classifier.predict_proba(data[features_trained])[:, 1].tolist())
            results['case:Y'].extend(len(data['case:Y_k'])*[case_y])
    results = pd.DataFrame.from_dict(results)
    fig = px.line(results, x='B', y='weight', color='label', facet_col='case:Y')
    fig.update_yaxes(showgrid=True, gridwidth=0.1, griddash='dash', gridcolor='black',
                     showline=True, linewidth=2, linecolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')
    fig.write_html(f'./figures/html/weight-functions.html')
    fig.write_image(f'./figures/svg/weight-functions.svg')
    fig.show()

