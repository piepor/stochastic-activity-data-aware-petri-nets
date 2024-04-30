import pm4py
import random
from utils.playout import apply_playout
from utils import sequence_data_aware_spn_model

log_name = 'sequence-data-aware-spn'
no_traces = 10000

def simulate(log_name, no_traces):
    # if log_name == 'data-aware-spn-paper':
    #     attributes = {'X': np.random.uniform(1, 10, no_traces), 'Y': random.choices(['k', 'l'], k=no_traces)}
    #     spn, im, fm, trans_prob = data_aware_spn_model.model()
    if log_name == 'sequence-data-aware-spn':
        attributes = {'Y': random.choices(['k', 'l'], k=no_traces)}
        spn, im, fm, trans_prob = sequence_data_aware_spn_model.model()
    else:
        raise NotImplementedError('Model not implemented')
    # spn, im, fm = create_stochastic_pn(False)
    pm4py.write_pnml(spn, im, fm, 'stochastic_petri_net.pnml')

    # breakpoint()
    log = apply_playout(spn, im, fm, trans_prob, log_name, attributes, no_traces=no_traces)
    pm4py.write_xes(log, f'logs/{log_name}.xes')
    return log
# df = pm4py.convert_to_dataframe(log)
# case_attr = (df
#              .drop_duplicates('case:concept:name')
#              .groupby('case:Y').count()
#              .reset_index()[['concept:name', 'case:Y']]
#              .rename(columns={'concept:name': 'count'}))
# fig = px.bar(case_attr, x='case:Y', y='count')
# fig.show()
# # breakpoint()
# # characteristics of log
# vars = pm4py.get_variants(df)
# vars_df = {'variant': [], 'count': []}
# for variant in vars:
#     vars_df['variant'].append("-".join(list(variant)))
#     vars_df['count'].append(vars[variant])
# vars_df = pd.DataFrame.from_dict(vars_df)
# fig = px.bar(vars_df, x='variant', y='count')
# fig.show()

if __name__ == '__main__':
    simulate(log_name, no_traces)
