from pm4py.objects.petri_net.semantics import ClassicSemantics
from utils import log_utils, plot_utils, training, evaluation, sequence_data_aware_spn_model

# WARNING "_" is used to determine the dummy part of the name: DO NOT USE ATTRIBUTES WITH "_"

def implementation(simulate=False):
    log, _ = log_utils.import_log('sequence-data-aware-spn', simulate=simulate)

    categorical_attrs = ['case:Y']
    net, im, fm, trans_prob_fns = sequence_data_aware_spn_model.model(False)

    semantic = ClassicSemantics()
    print("-----------------------------------\n")
    print("Training activity and data aware method...")
    print("\n-----------------------------------")
    classifiers_activities_data_aware, _, training_sets_activities_and_data, data_considered = training.train(log, net, im, fm, categorical_attrs)

    print("-----------------------------------\n")
    print("Training data aware method...")
    print("\n-----------------------------------")
    classifiers_data_aware, _, _, _ = training.train_data_aware(log, net, im, fm, categorical_attrs)
    classifiers = {'activity-data-aware': classifiers_activities_data_aware, 
                   'data-aware': classifiers_data_aware}
    classifiers['original'] = trans_prob_fns 

    print("-----------------------------------\n")
    print("Evaluation...")
    print("\n-----------------------------------")
    example_traces_probability = evaluation.compute_example_trace_probabilities(classifiers, categorical_attrs, net, im, fm, semantic)
    probabilities = evaluation.evaluate_example_states(net, im, fm, semantic, classifiers, categorical_attrs)
    duemscs_df = evaluation.compute_duemsc(log, classifiers, net, im, fm, data_considered, categorical_attrs)

    # PLOT
    print("-----------------------------------\n")
    print("Plotting...")
    print("\n-----------------------------------")
    plot_utils.plot_log_activities_stats(log)
    plot_utils.plot_petri_net(net, im, fm)
    plot_utils.plot_trace_probability(example_traces_probability)
    plot_utils.plot_weight_functions_sequence_version(
            classifiers_activities_data_aware, [('tau', None), ('C', 'C'), ('D', 'D')], training_sets_activities_and_data)  
    plot_utils.plot_activities_probability(probabilities)
    plot_utils.plot_duemsc(duemscs_df)


if __name__ == "__main__":
    implementation()
