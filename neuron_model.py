import cupy as cp


class NeuronNetwork:
    def __init__(self, num_neurons, min_connections=1000, max_connections=100000):
        self.num_neurons = num_neurons
        self.min_connections = min_connections
        self.max_connections = max_connections

        # 뉴런 상태 초기화
        self.threshold = cp.random.uniform(-50, -30, size=num_neurons)  # 역치
        self.membrane_potential = cp.zeros(num_neurons)  # 초기 막전위
        self.active_state = cp.zeros(num_neurons, dtype=bool)  # 활성화 상태

        # 시냅스 연결 초기화
        self.connections = cp.random.randint(0, num_neurons, (num_neurons, min_connections))
        self.weights = cp.random.uniform(0.1, 1.0, (num_neurons, min_connections))

    def stimulate_neurons(self, input_signals):
        """
        뉴런을 외부 신호로 자극.
        """
        self.membrane_potential += input_signals

    def update_neurons(self):
        """
        뉴런 상태를 업데이트: 탈분극, 재분극, 신호 전달.
        """
        # 역치를 초과한 뉴런을 활성화
        self.active_state = self.membrane_potential >= self.threshold

        # 활성화된 뉴런의 막전위를 초기화 (재분극)
        self.membrane_potential[self.active_state] = 0

        # 활성화된 뉴런이 신호를 전달
        outgoing_signals = cp.zeros_like(self.membrane_potential)
        outgoing_signals[self.active_state] = 1  # 활성화된 뉴런에서 신호 출력

        # 연결된 뉴런으로 신호 전달
        for i in range(self.connections.shape[1]):
            target_indices = self.connections[:, i]
            self.membrane_potential[target_indices] += outgoing_signals * self.weights[:, i]

    def update_weights(self):
        """
        Hebbian Learning을 기반으로 연결 강도를 업데이트.
        """
        active_indices = cp.where(self.active_state)[0]

        # 활성화된 뉴런 쌍의 가중치 증가
        for i in active_indices:
            for j in active_indices:
                if i != j:
                    connected_indices = self.connections[i] == j
                    self.weights[i, connected_indices] += 0.1  # 가중치 강화

        # 비활성화된 뉴런 연결 약화
        inactive_indices = cp.where(~self.active_state)[0]
        for i in inactive_indices:
            self.weights[i] *= 0.99  # 가중치 감소

    def prune_and_rewire(self, threshold=0.2):
        """
        약한 연결을 가지치고 새로운 연결을 형성.
        """
        for i in range(self.num_neurons):
            # 약한 연결 제거
            weak_connections = cp.where(self.weights[i] < threshold)[0]
            self.weights[i, weak_connections] = 0

            # 동적 연결 조정 (1천 ~ 10만 범위 유지)
            current_connections = len(self.connections[i])
            if current_connections < self.min_connections:
                # 부족한 연결 보충
                new_targets = cp.random.randint(0, self.num_neurons, self.min_connections - current_connections)
                new_weights = cp.random.uniform(0.1, 1.0, self.min_connections - current_connections)
                self.connections[i] = cp.append(self.connections[i], new_targets)
                self.weights[i] = cp.append(self.weights[i], new_weights)

            elif current_connections > self.max_connections:
                # 과도한 연결 제거
                indices_to_keep = cp.argsort(self.weights[i])[-self.max_connections:]
                self.connections[i] = self.connections[i][indices_to_keep]
                self.weights[i] = self.weights[i][indices_to_keep]

    def get_active_neurons(self):
        """
        활성화된 뉴런의 인덱스를 반환.
        """
        return cp.where(self.active_state)[0]
