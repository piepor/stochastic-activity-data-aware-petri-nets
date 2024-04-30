import os
import pandas as pd
import pm4py
from utils import general_utils, training, evaluation, log_properties, plot_utils
import warnings
warnings.filterwarnings('ignore')


def preparation(log_name, noise_threshold=0.8):
    log = pm4py.read_xes(f"./logs/{log_name}")
    log = general_utils.replace_underscore_in_columns(log)
    cases = list(log['case:concept:name'].unique())
    log = log[log['case:concept:name'].isin(cases)]
    log_prop = log_properties.LogProperties().get_log_properties(log_name.split(".xes")[0])
    log_train, log_test = training.split_log_train_test(log, train_perc=0.5)
    net, im, fm = pm4py.discover_petri_net_inductive(log, noise_threshold=noise_threshold)
    return log_train, log_test, log_prop, net, im, fm

def evaluate_method(log_train, log_test, log_prop, net, im, fm, method):
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
    # breakpoint()
    print("-----------------------------------\n")
    print("Computing duEMSC...")
    print("\n-----------------------------------")
    duemscs_df = evaluation.compute_duemsc(log_test, {method: classifiers}, net, im, fm, data_considered, 
                                           categorical_attrs=log_prop['categorical-attributes'], not_data_attrs=log_prop['not-data-attributes'])
    return duemscs_df['duemsc'].tolist()[0]

def logs_in_directory(logs_name):
    return len([log_name for log_name in logs_name if log_name in os.listdir("./logs")]) == len(logs_name)

def plot_results(results):
    plot_utils.plot_results(results)

def implementation(compute=False):
    if compute:
        log_names = ['BP12-AW.xes', 'bpic12-a.xes', 'BPIC2017-OfferLog.xes',
                     'bpic2020-DomesticDeclarations.xes', 'bpic2020-InternationalDeclarations.xes',
                     'bpic2020-RequestForPayment.xes', 'bpic2020-PrepaidTravelCost.xes']
        # log_names = ['bpic2020-PrepaidTravelCost.xes']
        assert logs_in_directory(log_names)
        methods = ['activity-data-aware', 'data-aware']
        results = {'model-type': [], 'log': [], 'duemsc': [], 'noise-threshold': []}
        noise_thresholds = [0.4, 0.6]
        for log_name in log_names:
            for i in range(3):
                for noise_threshold in noise_thresholds:
                    log_train, log_test, log_prop, net, im, fm = preparation(log_name, noise_threshold)
                    for method in methods:
                        print("===================================\n")
                        print(f"\nLog: {log_name} - Method: {method} - Noise threshold: {noise_threshold} - Sample: {i}")
                        print("\n\n===================================")
                        duemsc = evaluate_method(log_train, log_test, log_prop, net, im, fm, method)
                        results['model-type'].append(method)
                        results['log'].append(log_name)
                        results['duemsc'].append(duemsc)
                        results['noise-threshold'].append(noise_threshold)
        results_df = pd.DataFrame.from_dict(results)
        results_df.to_pickle('./results/results.pkl')
    else:
        results_df = pd.read_pickle('./results/results.pkl')
    plot_results(results_df)


if __name__ == "__main__":
    implementation()
