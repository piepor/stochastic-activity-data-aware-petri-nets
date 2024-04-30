import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.obj import Marking
from pm4py.objects.petri_net.utils import petri_utils
from typing import Tuple

def model(view: bool=False) -> Tuple[PetriNet, Marking, Marking, dict]:
    pn = PetriNet()

    trans_a = PetriNet.Transition('A', 'A')
    trans_b = PetriNet.Transition('B', 'B')
    trans_c = PetriNet.Transition('C', 'C')
    trans_d = PetriNet.Transition('D', 'D')
    trans_e = PetriNet.Transition('E', 'E')
    trans_f = PetriNet.Transition('F', 'F')
    trans_g = PetriNet.Transition('G', 'G')
    trans_h = PetriNet.Transition('H', 'H')
    trans_i = PetriNet.Transition('I', 'I')
    trans_l = PetriNet.Transition('L', 'L')
    trans_m = PetriNet.Transition('M')
    trans_n = PetriNet.Transition('N')

    source = PetriNet.Place('source')
    sink = PetriNet.Place('sink')
    p_1 = PetriNet.Place('p_1')
    p_2 = PetriNet.Place('p_2')
    p_3 = PetriNet.Place('p_3')
    p_4 = PetriNet.Place('p_4')
    p_5 = PetriNet.Place('p_5')
    p_6 = PetriNet.Place('p_6')
    p_7 = PetriNet.Place('p_7')
    p_8 = PetriNet.Place('p_8')

    pn.places.add(source)
    pn.places.add(p_1)
    pn.places.add(p_2)
    pn.places.add(p_3)
    pn.places.add(p_4)
    pn.places.add(p_5)
    pn.places.add(p_6)
    pn.places.add(p_7)
    pn.places.add(p_8)
    pn.places.add(sink)
    pn.transitions.add(trans_a)
    pn.transitions.add(trans_b)
    pn.transitions.add(trans_c)
    pn.transitions.add(trans_d)
    pn.transitions.add(trans_e)
    pn.transitions.add(trans_f)
    pn.transitions.add(trans_g)
    pn.transitions.add(trans_h)
    pn.transitions.add(trans_i)
    pn.transitions.add(trans_l)
    pn.transitions.add(trans_m)
    pn.transitions.add(trans_n)

    petri_utils.add_arc_from_to(source, trans_a, pn)
    petri_utils.add_arc_from_to(trans_a, p_1, pn)
    petri_utils.add_arc_from_to(p_1, trans_b, pn)
    petri_utils.add_arc_from_to(trans_b, p_2, pn)
    petri_utils.add_arc_from_to(trans_b, p_3, pn)
    petri_utils.add_arc_from_to(p_2, trans_d, pn)
    petri_utils.add_arc_from_to(p_3, trans_e, pn)
    petri_utils.add_arc_from_to(trans_d, p_5, pn)
    petri_utils.add_arc_from_to(trans_e, p_6, pn)
    petri_utils.add_arc_from_to(p_5, trans_m, pn)
    petri_utils.add_arc_from_to(trans_m, p_2, pn)
    petri_utils.add_arc_from_to(p_5, trans_h, pn)
    petri_utils.add_arc_from_to(p_6, trans_h, pn)
    petri_utils.add_arc_from_to(trans_h, p_8, pn)
    petri_utils.add_arc_from_to(p_1, trans_c, pn)
    petri_utils.add_arc_from_to(trans_c, p_4, pn)
    petri_utils.add_arc_from_to(p_4, trans_f, pn)
    petri_utils.add_arc_from_to(p_4, trans_g, pn)
    petri_utils.add_arc_from_to(trans_f, p_7, pn)
    petri_utils.add_arc_from_to(trans_g, p_7, pn)
    petri_utils.add_arc_from_to(p_7, trans_i, pn)
    petri_utils.add_arc_from_to(p_7, trans_n, pn)
    petri_utils.add_arc_from_to(trans_n, p_4, pn)
    petri_utils.add_arc_from_to(trans_i, p_8, pn)
    petri_utils.add_arc_from_to(p_8, trans_l, pn)
    petri_utils.add_arc_from_to(trans_l, sink, pn)
    
    # smap = {'A': 1, 'B': 0.6, 'C': 0.4, 'D': 0.7, 'E': 0.3, 'M': 0.5, 'H': 0.5,
    #         'F': 0.9, 'G': 0.1, 'I': 0.8, 'N': 0.2, 'L': 1}
    smap = {'A': 1, 'B': 0.6, 'C': 0.4, 'D': 0.7, 'E': 0.3, 'M': 0.5, 'H': 0.5,
            'F': 0.9, 'G': 0.1, 'I': 0.8, 'N': 0.2, 'L': 1}
    im = Marking()
    im[source] = 1
    fm = Marking()
    fm[sink] = 1
    if view:
        pm4py.view_petri_net(pn, im, fm)
    return pn, im, fm, smap

if __name__ == '__main__':
    model(True)
