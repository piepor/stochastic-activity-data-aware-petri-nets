import pm4py
# from pm4py.objects.petri_net.stochastic.obj import StochasticPetriNet
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.utils import petri_utils
from typing import Tuple

PERC_B = 0.4

def func_tau(num_b: int):
    return PERC_B/num_b

def func_c(num_b: int, attribute_y: str):
    if not isinstance(attribute_y, str):
        raise ValueError("Attribute Y must be numerical")
    if not attribute_y in ['k', 'l']:
        raise ValueError("Attribute Y must be 'k' or 'l'")
    return (1-PERC_B/num_b)*0.5 + (((PERC_B/num_b) / 4) * (attribute_y == 'k')) - (((PERC_B/num_b) / 4) * (attribute_y == 'l'))

def func_d(num_b: int, attribute_y: str):
    if not isinstance(attribute_y, str):
        raise ValueError("Attribute Y must be numerical")
    if not attribute_y in ['k', 'l']:
        raise ValueError("Attribute Y must be 'k' or 'l'")
    return (1-PERC_B/num_b)*0.5 - (((PERC_B/num_b) / 4) * (attribute_y == 'k')) + (((PERC_B/num_b) / 4) * (attribute_y == 'l'))

def model(view: bool=False) -> Tuple[PetriNet, Marking, Marking, dict]:
    spn = PetriNet("spn")

    trans_a = PetriNet.Transition('A', 'A')
    trans_b = PetriNet.Transition('B', 'B')
    trans_c = PetriNet.Transition('C', 'C')
    trans_d = PetriNet.Transition('D', 'D')
    tau = PetriNet.Transition('tau')

    smap = {'tau': func_tau, 'C': func_c, 'D': func_d}

    source = PetriNet.Place('source')
    sink = PetriNet.Place('sink')
    p_1 = PetriNet.Place('p_1')
    p_2 = PetriNet.Place('p_2')

    spn.places.add(source)
    spn.transitions.add(trans_a)
    spn.places.add(p_1)
    spn.transitions.add(trans_b)
    spn.places.add(p_2)
    spn.transitions.add(trans_c)
    spn.transitions.add(trans_d)
    spn.places.add(sink)
    spn.transitions.add(tau)

    petri_utils.add_arc_from_to(source, trans_a, spn)
    petri_utils.add_arc_from_to(trans_a, p_1, spn)
    petri_utils.add_arc_from_to(p_1, trans_b, spn)
    petri_utils.add_arc_from_to(trans_b, p_2, spn)
    petri_utils.add_arc_from_to(p_2, trans_c, spn)
    petri_utils.add_arc_from_to(p_2, trans_d, spn)
    petri_utils.add_arc_from_to(trans_c, sink, spn)
    petri_utils.add_arc_from_to(trans_d, sink, spn)
    petri_utils.add_arc_from_to(p_2, tau, spn)
    petri_utils.add_arc_from_to(tau, p_1, spn)

    im = Marking()
    im[source] = 1
    fm = Marking()
    fm[sink] = 1
    if view:
        pm4py.view_petri_net(spn, im, fm)
    return spn, im, fm, smap
