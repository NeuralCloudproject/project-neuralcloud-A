import numpy as np


class MemoryManager:
    """
    뉴런 네트워크의 기억 저장 및 검색 기능을 관리.
    """

    def __init__(self, neuron_network):
        self.network = neuron_network
        self.long_term_memory = {}  # 장기 기억 저장소

    def store_memory(self, input_text, active_neurons):
        """
        활성화된 뉴런 상태를 기억으로 저장.

        Args:
            input_text (str): 입력 텍스트.
            active_neurons (numpy array): 활성화된 뉴런의 인덱스.
        """
        pattern = tuple(active_neurons.tolist())
        self.long_term_memory[pattern] = input_text

    def recall_memory(self, active_neurons):
        """
        활성화된 뉴런 상태를 기반으로 기억 검색.

        Args:
            active_neurons (numpy array): 활성화된 뉴런의 인덱스.

        Returns:
            str: 검색된 기억 (없으면 None).
        """
        pattern = tuple(active_neurons.tolist())
        return self.long_term_memory.get(pattern, None)

    def consolidate_memory(self):
        """
        장기 기억을 통합하고 관련성을 강화.
        """
        for pattern in self.long_term_memory.keys():
            indices = np.array(pattern)
            for i in indices:
                for j in indices:
                    if i != j:
                        connected_indices = self.network.synapses[i] == j
                        self.network.synapses[i].weight += 0.05
