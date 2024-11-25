import cupy as cp
from synapse import Synapse


class NeuronNetwork:
    """
    연합뉴런 네트워크를 구현한 클래스. 뉴런 간 신호 전달 및 학습을 처리.
    """

    def __init__(self, num_neurons, min_connections=1000, max_connections=100000):
        """
        연합뉴런 네트워크 초기화.

        Args:
            num_neurons (int): 총 뉴런 수.
            min_connections (int): 각 뉴런의 최소 시냅스 연결 수.
            max_connections (int): 각 뉴런의 최대 시냅스 연결 수.
        """
        self.num_neurons = num_neurons
        self.min_connections = min_connections
        self.max_connections = max_connections

        # 뉴런 상태 초기화
        self.threshold = cp.random.uniform(-50, -50, size=num_neurons)  # 최소 역치값 -50
        self.membrane_potential = cp.full(num_neurons, -70.0)  # 초기 막전위 -70
        self.active_state = cp.zeros(num_neurons, dtype=bool)  # 활성화 상태
        self.action_potential_frequency = cp.random.uniform(1.0, 20.0, size=num_neurons)  # 활동전위 빈도
        self.action_potential_strength = cp.random.uniform(1.0, 5.0, size=num_neurons)  # 활동전위 강도

        # 시냅스 초기화
        self.synapses = self.initialize_synapses()

    def initialize_synapses(self):
        """
        시냅스 초기화: 뉴런 간 연결 생성.

        Returns:
            list: 초기화된 시냅스 리스트.
        """
        synapses = []
        for pre_neuron in range(self.num_neurons):
            num_connections = cp.random.randint(self.min_connections, self.max_connections)
            post_neurons = cp.random.randint(0, self.num_neurons, size=num_connections)
            weights = cp.random.uniform(0.1, 1.0, size=num_connections)
            vesicle_counts = cp.random.uniform(5, 100, size=num_connections)
            calcium_levels = cp.random.uniform(0.1, 1.0, size=num_connections)

            synapses.extend(
                [
                    Synapse(
                        pre_neuron,
                        post_neurons[i],
                        weights[i],
                        vesicle_counts[i],
                        calcium_levels[i],
                    )
                    for i in range(num_connections)
                ]
            )
        return synapses

    def stimulate_neurons(self, input_signals):
        """
        뉴런을 외부 신호로 자극.

        Args:
            input_signals (array-like): 외부 입력 신호.
        """
        self.membrane_potential += input_signals

    def update_neurons(self):
        """
        뉴런 상태 업데이트: 탈분극, 재분극, 신경전달물질 방출 및 신호 전달.
        """
        self.active_state = self.membrane_potential >= self.threshold
        self.membrane_potential[self.active_state] = -70  # 활성화 후 막전위 초기화

        for synapse in self.synapses:
            pre_active = self.active_state[synapse.pre_neuron]
            frequency = self.action_potential_frequency[synapse.pre_neuron]
            strength = self.action_potential_strength[synapse.pre_neuron]
            synapse.release_neurotransmitter(pre_active, frequency * strength)

            signal = synapse.transmit_signal()
            self.membrane_potential[synapse.post_neuron] += signal

    def update_weights(self):
        """
        Hebbian Learning 기반으로 시냅스 가중치 업데이트.
        """
        for synapse in self.synapses:
            pre_active = self.active_state[synapse.pre_neuron]
            post_active = self.active_state[synapse.post_neuron]
            synapse.update_weight(pre_active, post_active)
