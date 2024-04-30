import pm4py
import numpy as np

def compute_uemsc(traces_probabilities_log, traces_probabilities_model):
    for trace in traces_probabilities_log:
        if trace not in traces_probabilities_model:
            traces_probabilities_model[trace] = 0
    probabilities_diff = [traces_probabilities_log[trace] - traces_probabilities_model[trace] for trace in traces_probabilities_log]
    return 1-sum([max([prob_diff, 0]) for prob_diff in probabilities_diff])

def compute_emd(stochastic_language_x, stochastic_language_y):
    return pm4py.analysis.compute_emd(stochastic_language_x, stochastic_language_y)

def compute_duemsc(traces_probabilities_log, conditional_traces_probabilities_model, data_probabilities):
    traces_probabilities_model = list(map(lambda x, y: x*y, conditional_traces_probabilities_model, data_probabilities))
    probabilities_diff = np.array(traces_probabilities_log) - np.array(traces_probabilities_model)
    # breakpoint()
    return 1 - sum([max([prob_diff, 0]) for prob_diff in probabilities_diff])
