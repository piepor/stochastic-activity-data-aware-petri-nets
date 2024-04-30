from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.random_variables.random_variable import RandomVariable
from numpy.random import choice

def get_transitions_probabilities_sequence_data_aware(transitions_attrs_fns, attributes):
    trans_prob = {col: 0 for col in transitions_attrs_fns}
    for trans in trans_prob:
        if trans == 'tau':
            trans_prob[trans] = transitions_attrs_fns[trans](attributes['n_b'])
        if trans in ['C', 'D']:
            trans_prob[trans] = transitions_attrs_fns[trans](attributes['n_b'], attributes['Y'])
    trans_prob['A'] = 1
    trans_prob['B'] = 1
    return trans_prob

def get_transitions_probabilities_data_aware(transitions_attrs_fns, attributes):
    trans_prob = {col: 0 for col in transitions_attrs_fns}
    for attribute in attributes:
        if attribute == 'X':
            for trans in ['a', 'b']:
                trans_prob[trans] = transitions_attrs_fns[trans](attributes[attribute])
        if attribute == 'Y':
            for trans in ['c', 'd']:
                trans_prob[trans] = transitions_attrs_fns[trans](attributes[attribute])
    return trans_prob

def get_smap(net, transitions_probabilities):
    # breakpoint()
    smap = {}
    for trans in net.transitions:
        rand = RandomVariable()
        rand.read_from_string('IMMEDIATE', None)
        rand.set_weight(transitions_probabilities[trans.name])
        smap[trans] = rand
    return smap

def pick_transition(et, smap):
    """
    Pick a transition in a set of transitions based on the weights
    specified by the stochastic map
    Parameters
    --------------
    et
        Enabled transitions
    smap
        Stochastic map
    Returns
    --------------
    trans
        Transition chosen according to the weights
    """
    # breakpoint()
    wmap = {ct: smap[ct].get_weight() if ct in smap else 1.0 for ct in et}
    wmap_sv = sum(wmap.values())
    list_of_candidates = []
    probability_distribution = []
    # breakpoint()
    for ct in wmap:
        list_of_candidates.append(ct)
        if wmap_sv == 0:
            probability_distribution.append(1.0/float(len(wmap)))
        else:
            probability_distribution.append(wmap[ct] / wmap_sv)
    ct = list(choice(list_of_candidates, 1, p=probability_distribution))[0]
    return ct
