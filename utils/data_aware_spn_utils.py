import copy
from utils.data_alignments import group_data_step
from utils.general_utils import get_transition_by_name


def get_observation_points(data_alignments, log_model_align_map, net, im, semantic):
    observations_points = {}
    for alignment in data_alignments:
        marking = im
        log_align = log_model_align_map[alignment]
        grouped_data_step = group_data_step(data_alignments[alignment])
        for counter, fired_trans_name in enumerate(alignment):
            step_name = f"step-{counter}"
            enabled_trans = semantic.enabled_transitions(net, marking)
            grouped_data_step[step_name] = copy.copy(grouped_data_step[step_name])
            grouped_data_step_length = len(grouped_data_step[step_name][list(grouped_data_step[step_name].keys())[0]])
            # breakpoint()
            for trans in enabled_trans:
                if not trans.name in observations_points:
                    observations_points[trans.name] = {}
                for attr in grouped_data_step[step_name]:
                    if not attr in observations_points[trans.name]:
                        if len(observations_points[trans.name]) == 0:
                            observations_points[trans.name][attr] = copy.copy(grouped_data_step[step_name][attr])
                        else:
                            # if "fired" in the keys, it means that it is not the first time filling the dict:
                            # the number of "fired" is the right one since it is updated last and outside the loop.
                            if "fired" in observations_points[trans.name]:
                                dict_length = len(observations_points[trans.name]["fired"])
                                observations_points[trans.name][attr] = dict_length*[None]
                            else:
                                # if not "fired" in the keys, it means that it is the first time filling the dictionaries and we 
                                # do not have to add None values.
                                observations_points[trans.name][attr] = []
                            observations_points[trans.name][attr].extend(copy.copy(grouped_data_step[step_name][attr]))
                    else:
                        observations_points[trans.name][attr].extend(copy.copy(grouped_data_step[step_name][attr]))
                grouped_data_step_length = len(grouped_data_step[step_name][list(grouped_data_step[step_name].keys())[0]])
                # add transition fired
                if not "fired" in observations_points[trans.name]:
                    observations_points[trans.name]["fired"] = []
                observations_points[trans.name]["fired"].extend(grouped_data_step_length*[fired_trans_name])
                # add None vector to all data attributes not presente in the data of the step
                attr_not_present = set(observations_points[trans.name].keys()).difference(set(grouped_data_step[step_name].keys()))
                for attr in attr_not_present:
                    if attr != 'fired':
                        observations_points[trans.name][attr].extend(grouped_data_step_length*[None])
            # breakpoint()
            trans_to_fire = get_transition_by_name(net, fired_trans_name)
            marking = semantic.execute(trans_to_fire, net, marking) 
    return observations_points
