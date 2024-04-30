import time
import datetime
from copy import copy
from enum import Enum
from typing import Optional, Dict, Any, Union

from pm4py.objects import petri_net
from pm4py.objects.log import obj as log_instance
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.util import constants
from pm4py.util import exec_utils
from pm4py.util import xes_constants

from utils.stochastic_map import get_smap, pick_transition, get_transitions_probabilities_data_aware, get_transitions_probabilities_sequence_data_aware

class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY
    RETURN_VISITED_ELEMENTS = "return_visited_elements"
    NO_TRACES = "noTraces"
    MAX_TRACE_LENGTH = "maxTraceLength"
    LOG = "log"
    STOCHASTIC_MAP = "stochastic_map"
    PETRI_SEMANTICS = "petri_semantics"


def apply_playout(net, initial_marking: Marking, final_marking: Marking, transitions_probabilities_fn, model_type: str, attributes={}, no_traces=100, max_trace_length=100,
                  case_id_key=xes_constants.DEFAULT_TRACEID_KEY, activity_key=xes_constants.DEFAULT_NAME_KEY, timestamp_key=xes_constants.DEFAULT_TIMESTAMP_KEY,
                  semantics=petri_net.semantics.ClassicSemantics()) -> EventLog:
    """
    Do the playout of a Petrinet generating a log
    Parameters
    ----------
    net
        Petri net to play-out
    initial_marking
        Initial marking of the Petri net
    no_traces
        Number of traces to generate
    max_trace_length
        Maximum number of events per trace (do break)
    case_id_key
        Trace attribute that is the case ID
    activity_key
        Event attribute that corresponds to the activity
    timestamp_key
        Event attribute that corresponds to the timestamp
    final_marking
        If provided, the final marking of the Petri net
    smap
        Stochastic map
    log
        Log
    semantics
        Semantics of the Petri net to be used (default: petri_net.semantics.ClassicSemantics())
    """
    # from pm4py.algo.simulation.montecarlo.utils import replay
    # from pm4py.objects.stochastic_petri import utils as stochastic_utils

    curr_timestamp = time.time()
    all_visited_elements = []
    # attributes = {'x': np.random.uniform(1, 10, no_traces), 'y': random.choices(['k', 'l'], k=no_traces)}
    # y_values = random.choices(['k', 'l'], k=no_traces)
    for attr in attributes:
        if len(attributes[attr]) != no_traces:
            raise ValueError("Attributes number must be equal to the total number of traces")
    for i in range(no_traces):
        visited_elements = []
        visible_transitions_visited = []
        marking = copy(initial_marking)
        current_attributes = {attr: attributes[attr][i] for attr in attributes}
        if model_type == 'sequence-data-aware-spn':
            current_attributes['n_b'] = 0
        while len(visible_transitions_visited) < max_trace_length:
            # breakpoint()
            visited_elements.append(marking)

            if not semantics.enabled_transitions(net, marking):  # supports nets with possible deadlocks
                break
            all_enabled_trans = semantics.enabled_transitions(net, marking)
            if final_marking is not None and marking == final_marking:
                en_t_list = list(all_enabled_trans.union({None}))
            else:
                en_t_list = list(all_enabled_trans)
            # number_of_bs = sum(map(lambda trans: trans.label == 'B', visible_transitions_visited)) 
            # smap = get_smap(en_t_list, marking, y_values[i], number_of_bs)
            # breakpoint()
            if len(en_t_list) > 1:
                if model_type == 'data-aware-spn-paper':
                    transitions_probabilities = get_transitions_probabilities_data_aware(
                            transitions_probabilities_fn, current_attributes)
                elif model_type == 'sequence-data-aware-spn':
                    transitions_probabilities = get_transitions_probabilities_sequence_data_aware(
                            transitions_probabilities_fn, current_attributes)
                else:
                    raise NotImplemented('Model not implemented')
            else:
                transitions_probabilities = {trans.name: 1 for trans in net.transitions}
            smap = get_smap(net, transitions_probabilities)
            # breakpoint()
            trans = pick_transition(en_t_list, smap)

            if trans is None:
                break
            # breakpoint()
            if model_type == 'sequence-data-aware-spn' and trans.name == 'B':
                current_attributes['n_b'] += 1
            visited_elements.append(trans)
            if trans.label is not None:
                visible_transitions_visited.append(trans)

            marking = semantics.execute(trans, net, marking)
        # breakpoint()
        all_visited_elements.append(tuple(visited_elements))

    log = log_instance.EventLog()

    for index, visited_elements in enumerate(all_visited_elements):
        trace = log_instance.Trace()
        trace.attributes[case_id_key] = str(index)
        for attribute in attributes:
            trace.attributes[attribute] = attributes[attribute][index]
        # trace.attributes['Y'] = y_values[index]
        for element in visited_elements:
            if type(element) is StochasticPetriNet.Transition and element.label is not None:
                event = log_instance.Event()
                event[activity_key] = element.label
                event[timestamp_key] = datetime.datetime.fromtimestamp(curr_timestamp)
                trace.append(event)
                # increases by 1 second
                curr_timestamp += 1
        log.append(trace)
    return log


def apply(net: PetriNet, initial_marking: Marking, final_marking: Marking, attributes: dict, trans_probs_fn: dict={}, no_traces: int=1000,
          parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> EventLog:
    """
    Do the playout of a Petrinet generating a log
    Parameters
    -----------
    net
        Petri net to play-out
    initial_marking
        Initial marking of the Petri net
    final_marking
        If provided, the final marking of the Petri net
    parameters
        Parameters of the algorithm:
            Parameters.NO_TRACES -> Number of traces of the log to generate
            Parameters.MAX_TRACE_LENGTH -> Maximum trace length
            Parameters.PETRI_SEMANTICS -> Petri net semantics to be used (default: petri_nets.semantics.ClassicSemantics())
    """
    if parameters is None:
        parameters = {}
    case_id_key = exec_utils.get_param_value(Parameters.CASE_ID_KEY, parameters, xes_constants.DEFAULT_TRACEID_KEY)
    activity_key = exec_utils.get_param_value(Parameters.ACTIVITY_KEY, parameters, xes_constants.DEFAULT_NAME_KEY)
    timestamp_key = exec_utils.get_param_value(Parameters.TIMESTAMP_KEY, parameters,
                                               xes_constants.DEFAULT_TIMESTAMP_KEY)
    # no_traces = exec_utils.get_param_value(Parameters.NO_TRACES, parameters, 1000)
    max_trace_length = exec_utils.get_param_value(Parameters.MAX_TRACE_LENGTH, parameters, 1000)
    # smap = exec_utils.get_param_value(Parameters.STOCHASTIC_MAP, parameters, None)
    semantics = exec_utils.get_param_value(Parameters.PETRI_SEMANTICS, parameters, petri_net.semantics.ClassicSemantics())

    return apply_playout(net, initial_marking, final_marking, attributes=attributes, transitions_probabilities_fn=trans_probs_fn,
                         max_trace_length=max_trace_length, no_traces=no_traces, case_id_key=case_id_key, activity_key=activity_key,
                         timestamp_key=timestamp_key, semantics=semantics)
