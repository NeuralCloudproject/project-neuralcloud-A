import numpy as np


class MemoryManager:
    def __init__(self, neuron_network):
        self.network = neuron_network
        self.long_term_memory = {}

    def store_memory(self, input_text, active_neurons):
        """
        활성화된 뉴런 상태를 장기 기억으로 저장.
        """
        pattern = tuple(active_neurons.tolist())
        self.long_term_memory[pattern] = input_text

    def recall_memory(self, active_neurons):
        """
        활성화된 뉴런 상태를 기반으로 기억 검색.
        """
        pattern = tuple(active_neurons.tolist())
        return self.long_term_memory.get(pattern, None)

    def consolidate_memory(self):
        """
        장기 기억 패턴 간의 연결 강화를 통해 통합.
        """
        for pattern in self.long_term_memory.keys():
            indices = np.array(pattern)
            for i in indices:
                for j in indices:
                    if i != j:
                        connected_indices = self.network.connections[i] == j
                        self.network.weights[i, connected_indices] += 0.05
