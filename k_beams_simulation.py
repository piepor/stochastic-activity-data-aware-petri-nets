import logging
import warnings
from queue import Queue
from pm4py.objects import petri_net
from copy import copy

semantics = petri_net.semantics.ClassicSemantics()

def get_visible_sequence(sequence):
    return [trans.label for trans in sequence if trans.label]

def get_transitions_probabilities(enabled_transitions, smap):
    weight_trans = {}
    for trans in enabled_transitions:
        weight_trans[trans.name] = get_weight(trans, smap)
    return {trans: weight_trans[trans]/sum(weight_trans.values()) for trans in weight_trans}

def get_weight(trans, smap):
    return smap[trans.name]

class Memory:
    def __init__(self):
        self.map_sequences_markings = {}
        self.sequences_probability = {}
        self.map_sequences = {}

    def get_most_probable_sequence(self):
        seq_prob_tuples = sorted(self.sequences_probability.items(), key= lambda x: x[1], reverse=True)
        return seq_prob_tuples[0][0]

    def get_marking_most_probable_sequence(self):
        most_prob_seq = self.get_marking_most_probable_sequence()
        return copy(self.map_sequences_markings[most_prob_seq])

    def add_sequence(self, sequence, prob, marking):
        self.map_sequences_markings[str(sequence)] = marking
        self.sequences_probability[str(sequence)] = prob
        self.map_sequences[str(sequence)] = copy(sequence)

    def get_sequences(self):
        return copy(list(self.sequences_probability.keys()))

    def get_sequences_more_probable_than_threshold(self, threshold):
        return copy(dict(filter(lambda x: x[1] >= threshold, self.sequences_probability.items())))

    def get_map_from_sequences(self, sequences):
        return copy(dict(filter(lambda x: x[0] in sequences, self.map_sequences_markings.items())))

    def get_sequences_from_sequences_name(self, sequences):
        return copy(dict(filter(lambda x: x[0] in sequences, self.map_sequences.items())))

    def filter_sequences_on_threshold(self, threshold):
        sequences_probability = self.get_sequences_more_probable_than_threshold(threshold)
        sequences = self.get_sequences_from_sequences_name(sequences_probability.keys())
        markings = self.get_map_from_sequences(sequences_probability.keys())
        return copy(sequences_probability), copy(sequences), copy(markings)

    def delete_sequences(self, sequences):
        sequences_probability = copy(self.sequences_probability)
        self.sequences_probability = {sequence: sequences_probability[sequence] for sequence in sequences_probability if sequence not in sequences}
        map_sequences = copy(self.map_sequences)
        self.map_sequences = {sequence: map_sequences[sequence] for sequence in map_sequences if sequence not in sequences} 
        map_sequences_markings = copy(self.map_sequences_markings)
        self.map_sequences_markings = {sequence: map_sequences_markings[sequence] for sequence in map_sequences_markings if sequence not in sequences}

    def get_probability_visible_sequences(self):
        visible_sequences = {str(get_visible_sequence(self.map_sequences[path])): [] for path in self.map_sequences}
        for path in self.sequences_probability:
            visible_sequences[str(get_visible_sequence(self.map_sequences[path]))].append(self.sequences_probability[str(path)])
        return copy({visible_sequence: sum(visible_sequences[visible_sequence]) for visible_sequence in visible_sequences})

    def filter_sequences_on_visible_sequences(self, visible_sequences):
        sequences_probability = self.get_paths_from_visible_sequences(visible_sequences)
        sequences = self.get_sequences_from_sequences_name(sequences_probability.keys())
        markings = self.get_map_from_sequences(sequences_probability.keys())
        return copy(sequences_probability), copy(sequences), copy(markings)

    def get_paths_from_visible_sequences(self, visible_sequences):
        sequences_probability = copy(self.sequences_probability)
        return {path: sequences_probability[path] for path in sequences_probability if str(get_visible_sequence(self.map_sequences[path])) in visible_sequences}

    def get_lowest_visible_sequence_probability(self):
        sequences_probability = self.get_probability_visible_sequences()
        lowest = 0
        if sequences_probability:
            lowest = min(sequences_probability.values()) 
        return lowest


class Tracker:
    def __init__(self, num_beams, final_marking):
        self.num_beams = num_beams
        self.final_marking = final_marking
        self.map_markings_sequences = {}
        self.sequences_probability = {}
        self.visible_sequences_map = {}
        self.top_k_visible_sequences = {}
        self.paths_explored = []

    def get_sequences_marking(self, marking):
        sequences_marking = []
        if marking in self.map_markings_sequences:
            sequences_marking = copy(self.map_markings_sequences[marking])
        return sequences_marking

    def update(self, old_sequence, trans, prob_trans):
        new_sequence = copy(old_sequence)
        new_sequence.append(trans)
        self.add_path(new_sequence)
        self.update_sequences_probability(old_sequence, prob_trans, trans)
        self.update_visible_sequences_map(new_sequence)
        self.update_top_k_visible()

    def update_sequences_probability(self, sequence, probability, trans):
        new_sequence = copy(sequence)
        new_sequence.append(copy(trans))
        if str(sequence) in self.sequences_probability:
            self.sequences_probability[str(new_sequence)] = copy(self.sequences_probability[str(sequence)]*probability)
        else:
            self.sequences_probability[str(new_sequence)] = probability

    def add_path(self, path):
        self.paths_explored.append(path)

    def update_sequences(self, memory: Memory):
        # paths explored without the paths offloaded to memory
        sequences = copy(self.paths_explored)
        self.paths_explored = [sequence for sequence in sequences if str(sequence) not in memory.get_sequences()]
        sequences_probability = copy(self.sequences_probability)
        self.sequences_probability = {str(sequence): sequences_probability[str(sequence)] for sequence in self.paths_explored}
        visible_paths_explored = [str(get_visible_sequence(sequence)) for sequence in self.paths_explored]
        visible_sequences_map = copy(self.visible_sequences_map)
        self.visible_sequences_map = {
                sequence: visible_sequences_map[sequence] for sequence in visible_sequences_map if sequence in visible_paths_explored}
        top_k = copy(self.top_k_visible_sequences)
        valid_visible_sequences = [str(get_visible_sequence(sequence)) for sequence in self.paths_explored]
        self.top_k_visible_sequences = [sequence for sequence in top_k  if sequence in valid_visible_sequences]

    def delete_less_probable_sequences(self):
        old_top_k = copy(self.get_top_k())
        new_top_k = old_top_k[:self.num_beams]
        # update the other quantities with references to only the 
        self.top_k_visible_sequences = new_top_k

    def update_visible_sequences_map(self, sequence):
        if str(get_visible_sequence(sequence)) not in self.visible_sequences_map:
            self.visible_sequences_map[str(get_visible_sequence(sequence))] = []
        self.visible_sequences_map[str(get_visible_sequence(sequence))].append(copy(sequence))

    def get_top_k_probability(self):
        top_k_visible_sequences_prob = {}
        for visible_sequence in self.top_k_visible_sequences:
            sequence_probability = sum([self.sequences_probability[str(sequence)] for sequence in self.visible_sequences_map[visible_sequence]])
            top_k_visible_sequences_prob[visible_sequence] = copy(sequence_probability)
        return top_k_visible_sequences_prob

    def update_top_k_visible(self):
        # print(f"VISIBLE SEQUENCE MAP: {self.visible_sequences_map}")
        # print(f"SEQUENCE PROBABILITY: {self.sequences_probability}")
        visible_sequences_prob = {}
        for visible_sequence in self.visible_sequences_map:
            sequence_probability = sum([self.sequences_probability[str(sequence)] for sequence in self.visible_sequences_map[visible_sequence]])
            visible_sequences_prob[visible_sequence] = copy(sequence_probability)
        seq_prob_tuples = sorted(visible_sequences_prob.items(), key= lambda x: x[1], reverse=True)
        self.top_k_visible_sequences = [seq_prob[0] for seq_prob in seq_prob_tuples]

    def update_map_markings_sequences(self, marking, sequences):
        if not marking in self.map_markings_sequences:
            self.map_markings_sequences[marking] =  copy(sequences)
        else:
            self.map_markings_sequences[marking].extend(copy(sequences))

    def delete_marking_paths(self, marking):
        paths_explored = copy(self.paths_explored)
        if marking in self.map_markings_sequences:
            marking_paths = self.map_markings_sequences[marking]
            self.paths_explored = copy([path for path in paths_explored if path not in marking_paths])
            # using a set because if different sequences have the same visible one, they will be contained
            # at the same key in the visible sequences map
            visible_markings_paths = {str(get_visible_sequence(marking_path)) for marking_path in marking_paths}
            for visible_marking_path in visible_markings_paths:
                old_sequences = copy(self.visible_sequences_map[visible_marking_path])
                new_sequences = [path for path in old_sequences if path not in marking_paths]
                if new_sequences:
                    self.visible_sequences_map[visible_marking_path] = copy(new_sequences)
                else:
                    visible_map = copy(self.visible_sequences_map)
                    self.visible_sequences_map = {sequence: visible_map[sequence] for sequence in visible_map if not str(sequence) == visible_marking_path}
        
    def reset_map_markings_sequences(self, marking):
        self.map_markings_sequences[marking] = []

    def get_top_k(self):
        return self.top_k_visible_sequences

    def get_sequences_probability(self):
        return self.sequences_probability

    def remove_ended_sequence(self, sequence_to_remove):
        # breakpoint()
        top_k = copy(self.top_k_visible_sequences)
        self.top_k_visible_sequences = [sequence for sequence in top_k if sequence != str(get_visible_sequence(sequence_to_remove))]
        paths_explored = copy(self.paths_explored)
        self.paths_explored = [sequence for sequence in paths_explored if sequence != sequence_to_remove]
        visible_sequences_map = copy(self.visible_sequences_map)
        visible_sequence_to_remove = copy(get_visible_sequence(sequence_to_remove))
        self.visible_sequences_map[str(visible_sequence_to_remove)] = [sequence for sequence in visible_sequences_map if sequence != sequence_to_remove]
        sequences_probability = copy(self.sequences_probability)
        self.sequences_probability = {sequence: sequences_probability[sequence] for sequence in sequences_probability if sequence != str(sequence_to_remove)}

    # def check_the_memory_for_others_top_k(self, memory):
    #     sequences_probability = copy(self.sequences_probability)
    #     lowest = min(sequences_probability.values())
    #     return memory.filter_sequences_on_threshold(lowest)

    def check_the_memory_for_others_top_k(self, memory: Memory):
        seq_prob = copy(memory.get_probability_visible_sequences())
        lowest_tracker = 0
        if self.sequences_probability:
            lowest_tracker = min(self.sequences_probability.values())
        selected_sequences = {sequence: seq_prob[sequence] for sequence in seq_prob if seq_prob[sequence] >= lowest_tracker}
        return memory.filter_sequences_on_visible_sequences(selected_sequences)

    def send_to_memory_the_others_sequences(self, memory: Memory):
        for path in self.paths_explored:
            visible_sequence = get_visible_sequence(path)
            if not str(visible_sequence) in self.get_top_k():
                sequence_marking = [marking for marking in self.map_markings_sequences if path in self.map_markings_sequences[marking]][0]
                memory.add_sequence(path, self.sequences_probability[str(path)], sequence_marking)
        self.update_sequences(memory)
        for marking in self.map_markings_sequences:
            paths = copy(self.map_markings_sequences[marking])
            self.map_markings_sequences[marking] = [path for path in paths if path in self.paths_explored]

    def recover_paths_from_memory(self, memory: Memory, regulator):
        seq_prob_mem, seq_mem, mark_mem = self.check_the_memory_for_others_top_k(memory)
        memory.delete_sequences(seq_prob_mem.keys())
        if seq_prob_mem:
            paths_memory = [seq_mem[sequence] for sequence in seq_mem]
            self.paths_explored.extend(paths_memory)
            for sequence in seq_prob_mem:
                self.sequences_probability[sequence] = copy(seq_prob_mem[sequence])
                visible_sequence = str(get_visible_sequence(seq_mem[sequence]))
                if visible_sequence in self.visible_sequences_map:
                    self.visible_sequences_map[visible_sequence].append(copy(seq_mem[sequence]))
                else:
                    self.visible_sequences_map[visible_sequence] = [copy(seq_mem[sequence])]
                marking = mark_mem[sequence]
                if marking in self.map_markings_sequences:
                    self.map_markings_sequences[marking].append(copy(seq_mem[sequence]))
                else:
                    self.map_markings_sequences[marking] = [copy(seq_mem[sequence])]
        self.update_top_k_visible()
        self.delete_less_probable_sequences()
        self.send_to_memory_the_others_sequences(memory)
        # add marking of the sequences recovered
        markings_recovered = [mark_mem[path] for path in mark_mem if seq_mem[path] in self.paths_explored]
        # print(f"MARKINGS RECOVERED: {markings_recovered}")
        # print(f"Marking Memory: {mark_mem}")
        if markings_recovered:
            for marking in markings_recovered:
                regulator.add_marking(marking)

    def get_lowest_visible_sequence_probability(self):
        lowest = 0
        if self.sequences_probability:
            lowest = min(self.sequences_probability.values()) 
        return lowest


class Regulator:
    def __init__(self, final_marking, num_beams, max_length=1000):
        self.markings = Queue()
        self.final_marking = copy(final_marking)
        self.max_length = max_length
        self.num_beams = num_beams
        self.top_k_ended = {}
        self.ended_sequences = {}

    def add_marking(self, marking):
        self.markings.put(marking)

    def check_ended_sequence_probability(self, tracker: Tracker, memory: Memory):
        lowest_tracker = tracker.get_lowest_visible_sequence_probability()
        lowest_memory = memory.get_lowest_visible_sequence_probability()
        lowest_ended = 0
        if self.top_k_ended:
            lowest_ended = min(self.top_k_ended.values())
        return lowest_ended > lowest_tracker and lowest_ended > lowest_memory

    def continue_beam_search(self, tracker, memory):
        not_empty = copy(not self.markings.empty())
        top_k_ended_sequence_are_most_probable = self.check_ended_sequence_probability(tracker, memory)
        top_k_ended_sequence_reached = len(self.top_k_ended) >= tracker.num_beams
        go_on = not_empty and not (top_k_ended_sequence_are_most_probable and top_k_ended_sequence_reached)
        logging.debug(f"NOT EMPTY: {not_empty}")
        logging.debug(f"TOP K ENDED SEQUENCES ARE MOST PROBABLE: {top_k_ended_sequence_are_most_probable}")
        logging.debug(f"TOP K ENDED SEQUENCE REACHED: {top_k_ended_sequence_reached}")
        return copy(go_on)

    def get_marking(self):
        return self.markings.get()

    def check_marking_post(self, marking, sequences, tracker: Tracker):
        if marking == self.final_marking:
            # breakpoint()
            for sequence in sequences:
                visible_sequence = get_visible_sequence(sequence)
                if str(visible_sequence) not in self.ended_sequences:
                    self.ended_sequences[str(visible_sequence)] = 0
                self.ended_sequences[str(visible_sequence)] += tracker.sequences_probability[str(sequence)]
                tracker.remove_ended_sequence(sequence)
            self.update_top_k_ended()
        elif sequences:
            # new marking in queue
            self.markings.put(marking)

    def check_sequence(self, sequence, top_k_visible_sequences):
        go_on = False
        if str(get_visible_sequence(sequence)) in top_k_visible_sequences:
            go_on = True
        return go_on

    def check_marking_sequences(self, sequences, top_k_visible_sequences):
        go_on = False
        for sequence in sequences:
            if str(get_visible_sequence(sequence)) in top_k_visible_sequences:
                go_on = True
        return go_on

    def check_length(self, sequence):
        return len(sequence) <= self.max_length

    def update_top_k_ended(self):
        ended_sequences = copy(self.ended_sequences)
        if len(ended_sequences) > self.num_beams:
            # breakpoint()
            top_k_ended_new = sorted(ended_sequences.items(), key= lambda x: x[1], reverse=True)
            top_k_ended_new = [seq_prob[1] for seq_prob in top_k_ended_new][:self.num_beams]
            # check if there are others sequences with the same probability
            lowest = top_k_ended_new[-1]
            self.top_k_ended = dict(filter(lambda x: x[1] >= lowest, ended_sequences.items()))
        else:
            self.top_k_ended = ended_sequences


def simulate(net, initial_marking, final_marking, smap, num_beams):
    tracker = Tracker(num_beams, final_marking)
    regulator = Regulator(final_marking, num_beams)
    memory = Memory()
    marking = copy(initial_marking)
    regulator.add_marking(marking)
    # breakpoint()
    while regulator.continue_beam_search(tracker, memory):
        logging.debug(f"{90*'='}\n\n")
        marking = regulator.get_marking()
        if not semantics.enabled_transitions(net, marking):  # supports nets with possible deadlocks
            warnings.warn("Deadlock reached") 
            continue
        all_enabled_trans = semantics.enabled_transitions(net, marking)
        # breakpoint()
        prob_trans = get_transitions_probabilities(all_enabled_trans, smap)
        sequences_marking = tracker.get_sequences_marking(marking)
        # check if the marking is connected to a top k visible sequence
        if not regulator.check_marking_sequences(sequences_marking, tracker.get_top_k()) and not marking == initial_marking:
            # breakpoint()
            logging.debug(f"Sequences {sequences_marking} of marking {marking} not contain top - k  visible sequences")
            for sequence in sequences_marking:
                sequence_probability = copy(tracker.sequences_probability[str(sequence)])
                memory.add_sequence(sequence, sequence_probability, marking)
            tracker.delete_marking_paths(marking)
            tracker.reset_map_markings_sequences(marking)
            continue
        for trans in all_enabled_trans:
            new_sequences = []
            if marking == initial_marking and not tracker.get_top_k():
                new_sequence = copy([trans])
                new_sequences.append(new_sequence)
                tracker.update([], trans, prob_trans[trans.name])
            for sequence in sequences_marking:
                # check if the considered sequence is connected to a top k visible sequence
                new_sequence = copy(sequence)
                new_sequence.append(trans)
                # add sequence to the new sequences or to the memory only if the new sequence 
                # shorter than the maximum length
                if regulator.check_length(new_sequence):
                    if regulator.check_sequence(sequence, tracker.get_top_k()):
                        new_sequences.append(new_sequence)
                        tracker.update(sequence, trans, prob_trans[trans.name])
                    else:
                        sequence_probability = copy(tracker.sequences_probability[str(sequence)])
                        memory.add_sequence(sequence, sequence_probability, marking)
            new_marking = semantics.execute(trans, net, marking)
            tracker.update_map_markings_sequences(new_marking, new_sequences)
            regulator.check_marking_post(new_marking, new_sequences, tracker)
        tracker.delete_marking_paths(marking)
        tracker.reset_map_markings_sequences(marking)
        tracker.update_sequences(memory)
        tracker.update_top_k_visible()
        tracker.delete_less_probable_sequences()
        # breakpoint()
        tracker.recover_paths_from_memory(memory, regulator)
        ### DEBUG ###
        logging.debug(f"{30*'^'}")
        logging.debug(f"TOP-k: {tracker.top_k_visible_sequences}")
        logging.debug(f"{30*'^'}\n")
        logging.debug(f"Sequences Probability: {tracker.sequences_probability} \n")
        logging.debug(f"Visible Sequences Map: {tracker.visible_sequences_map} \n")
        logging.debug(f"Map Markings - Sequences: {tracker.map_markings_sequences} \n")
        logging.debug(f"Current Marking: {marking}")
        logging.debug(f"Markings Queue: {regulator.markings.queue} \n")
        logging.debug(f"Ended Sequences: {regulator.ended_sequences}")
        logging.debug(f"Top K Ended Sequences: {regulator.top_k_ended}\n")
        logging.debug(f"{30*'-'}")
        logging.debug(f"Paths explored: {tracker.paths_explored}")
        logging.debug(f"{30*'-'}\n")
        logging.debug(f"{30*'*'}")
        logging.debug(f"Memory: {memory.sequences_probability}")
        logging.debug(f"{30*'*'}")
        logging.debug(f"\n{90*'='}\n\n")
        # breakpoint()
    logging.debug(f"Considered top-k sequences: {tracker.get_top_k()}\n")
    logging.debug(f"\n{90*'X'}\n\n")
    logging.debug(f"Top-k ended sequences with maximum length {regulator.max_length}: {regulator.top_k_ended}")
    logging.debug(f"\n{90*'X'}\n\n")
    return regulator.top_k_ended


if __name__ == '__main__':
    import example_net
    net, im, fm, smap = example_net.model()
    logging.basicConfig(level=logging.DEBUG)
    top_k_ended = simulate(net, im, fm, smap, 5)
