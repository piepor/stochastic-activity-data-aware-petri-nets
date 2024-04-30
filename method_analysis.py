import pandas as pd
import pm4py
from utils import general_utils, log_properties, training, evaluation, plot_utils

def preparation(log_name, noise_threshold=0.8):
    log = pm4py.read_xes(f"./logs/{log_name}")
    log = general_utils.replace_underscore_in_columns(log)
    cases = list(log['case:concept:name'].unique())
    log = log[log['case:concept:name'].isin(cases)]
    log_prop = log_properties.LogProperties().get_log_properties(log_name.split(".xes")[0])
    log_train, log_test = training.split_log_train_test(log, train_perc=0.5)
    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=noise_threshold)
    return log_train, log_test, log_prop, net, im, fm

def train_classifiers(log_train, log_prop, net, im, fm, method):
    if method == 'activity-data-aware':
        train_fn = training.train
    elif method == 'data-aware':
        train_fn = training.train_data_aware
    else:
        raise NotImplementedError(f"Method {method} not implemented.")
    print("-----------------------------------\n")
    print("Training...")
    print("\n-----------------------------------")
    classifiers, _, _, data_considered = train_fn(log_train, net, im, fm,
                                                  categorical_attrs=log_prop['categorical-attributes'], not_data_attrs=log_prop['not-data-attributes'])
    return classifiers, data_considered

def implementation(compute=False):
    if compute:
        # log_names = ['BPIC2017-OfferLog.xes', 'bpic2020-PrepaidTravelCost.xes']
        log_names = ['bpic2020-PrepaidTravelCost.xes']
        noise_thresholds = [0.4, 0.6, 0.8]
        methods = ['activity-data-aware', 'data-aware']
        results = pd.DataFrame()
        for log_name in log_names:
            for noise_threshold in noise_thresholds:
                log_train, log_test, log_prop, net, im, fm = preparation(log_name, noise_threshold)
                pm4py.view_petri_net(net, im, fm)
                pm4py.vis.save_vis_petri_net(net, im, fm, file_path=f'./figures/{log_name}-{noise_threshold}-petri-net.svg')
                for method in methods:
                    print("===================================\n")
                    print(f"\nLog: {log_name} - Method: {method} - Noise threshold: {noise_threshold}")
                    print("\n\n===================================")
                    classifiers, data_considered = train_classifiers(log_train, log_prop, net, im, fm, method)
                    duemscs_df = evaluation.compute_duemsc_for_analysis(log_test, {method: classifiers}, net, im, fm, data_considered, 
                                                                        categorical_attrs=log_prop['categorical-attributes'], not_data_attrs=log_prop['not-data-attributes'])
                    duemscs_df.insert(len(duemscs_df.columns), 'noise-threshold', str(noise_threshold))
                    duemscs_df.insert(len(duemscs_df.columns), 'log', log_name)
                    results = pd.concat([results, duemscs_df])
        results.to_pickle('./results/analysis_results.pkl')
    else:
        results = pd.read_pickle('./results/analysis_results.pkl')
    plot_utils.plot_analysis_results(results)


if __name__ == "__main__":
    implementation()
