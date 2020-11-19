import torch

from itertools import count
from queue import PriorityQueue


class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """
    def __init__(self, beam_size, max_len, pad):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.nodes = PriorityQueue() # beams to be expanded
        self.final = PriorityQueue() # beams that ended in EOS

        self._counter = count() # for correct ordering of nodes with same score

    def add(self, score, node):
        """ Adds a new beam search node to the queue of current nodes """
        self.nodes.put((score, next(self._counter), node))

    def add_final(self, score, node):
        """ Adds a beam search path that ended in EOS (= finished sentence) """
        # ensure all node paths have the same length for batch ops
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        self.final.put((score, next(self._counter), node))

    def diverse_best(self, tgt_dict, gamma):
        """ Punishes bottom-ranked hypotheses stemming from same parent hypothesis to get more diverse N-best list. """
        reranked_nodes = PriorityQueue()
        reranked_final_nodes = PriorityQueue()

        # this dict stores parent beam in key and indices of all child beams (w.r.t. nodes_list) in list as values
        same_parents_hypotheses = dict()
        nodes_list = []

        for i in range(self.nodes.qsize()):
            node = self.nodes.get()
            nodes_list.append(node)
            # if parent beam (all tokens up to last) has been unseen
            if str(node[2].sequence[:-1]) not in same_parents_hypotheses:
                lst = [i]
                same_parents_hypotheses[str(node[2].sequence[:-1])] = lst
            # if parent beam has been seen, append nodes_list index to value of parent beam key in same_parents_hypotheses dict
            else:
                same_parents_hypotheses[str(node[2].sequence[:-1])].append(i)

        sequence_length = nodes_list[0][2].length

        # also look at finished sentences, but only if their <EOS> is at same time step as other generated words
        if not self.final.empty():
            for i in range(len(nodes_list), len(nodes_list) + self.final.qsize()):
                node = self.final.get()
                nodes_list.append(node)

                # only consider ended sentences when we're at timestep of sequence_length
                if node[2].sequence[sequence_length] == tgt_dict.eos_idx:  # <EOS> token
                    if str(node[2].sequence[:sequence_length]) not in same_parents_hypotheses:
                        lst = [i]
                        same_parents_hypotheses[str(node[2].sequence[:sequence_length])] = lst
                    else:
                        same_parents_hypotheses[str(node[2].sequence[:sequence_length])].append(i)

                else:
                    reranked_final_nodes.put(node)

        # go through sentences with same parent beams
        for parent, indices in same_parents_hypotheses.items():
            child_nodes = PriorityQueue()
            for index in indices:
                child_nodes.put(nodes_list[index])

            # again use property of PriorityQueues that smallest (most probable) values are retrieved first
            for i in range(child_nodes.qsize()):
                child_node = list(child_nodes.get())
                child_node[0] = child_node[0] + i * gamma  # the larger i and gamma, the more the translation is punished if it has a better sibling
                child_node = tuple(child_node)
                if self.pad in child_node[2].sequence:
                    reranked_final_nodes.put(child_node)
                else:
                    reranked_nodes.put(child_node)

        self.nodes = reranked_nodes
        self.final = reranked_final_nodes

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_best(self, alpha, n_best):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = PriorityQueue()
        merged_normalised = PriorityQueue()

        for _ in range(self.final.qsize()):
            node = self.final.get()
            merged.put(node)

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        # length normalisation
        for _ in range(merged.qsize()):
            node = merged.get()
            sent_length = node[2].length
            lp = (5 + sent_length) ** alpha / 6 ** alpha
            node_list = list(node)
            node_list[0] = node_list[0] / lp
            node = tuple(node_list)
            merged_normalised.put(node)

        # return n best translations
        nodes_list = []
        for _ in range(n_best):
            node = merged_normalised.get()
            node = (node[0], node[2])
            nodes_list.append(node)

        return nodes_list

    def prune(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """
        nodes = PriorityQueue()
        # Keep track of how many search paths are already finished (EOS)
        finished = self.final.qsize()
        for _ in range(self.beam_size-finished):
            node = self.nodes.get()
            nodes.put(node)
        self.nodes = nodes


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""
    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length):

        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search

    def eval(self):
        """ Returns score of sequence up to this node """
        return self.logp
