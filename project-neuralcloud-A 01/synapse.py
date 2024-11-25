class Synapse:
    """
    현실의 시냅스 작동 방식을 모방한 클래스.
    """

    def __init__(self, pre_neuron, post_neuron, weight, vesicle_count, calcium_concentration):
        """
        시냅스 초기화.

        Args:
            pre_neuron (int): 프리뉴런 ID.
            post_neuron (int): 포스트뉴런 ID.
            weight (float): 시냅스 가중치.
            vesicle_count (float): 시냅스 소포 수.
            calcium_concentration (float): 칼슘 이온 농도.
        """
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.neurotransmitter_level = 0.0  # 신경전달물질 농도

        # 생물학적 파라미터
        self.vesicle_count = vesicle_count  # 시냅스 소포의 수
        self.calcium_concentration = calcium_concentration  # 칼슘 이온 농도
        self.receptor_sensitivity = 1.0  # 수용체 민감도
        self.feedback = 0.0  # 후시냅스에서 전달되는 피드백

    def release_neurotransmitter(self, action_potential_frequency, action_potential_strength):
        """
        활동전위 강도와 빈도를 기반으로 신경전달물질 방출량 계산.

        Args:
            action_potential_frequency (float): 활동전위 빈도.
            action_potential_strength (float): 활동전위 강도.
        """
        calcium_effect = self.calcium_concentration * self.vesicle_count
        release_amount = (
            self.weight
            * action_potential_frequency
            * action_potential_strength
            * calcium_effect
        )
        self.neurotransmitter_level += release_amount

    def transmit_signal(self):
        """
        신경전달물질을 기반으로 신호 전달.

        Returns:
            float: 포스트뉴런으로 전달되는 신호 크기.
        """
        signal = self.neurotransmitter_level * self.receptor_sensitivity
        self.neurotransmitter_level *= 0.95  # 신경전달물질의 자연 감소
        return signal

    def update_feedback(self, feedback_value):
        """
        후시냅스 뉴런에서 전달된 피드백에 따라 민감도 조정.

        Args:
            feedback_value (float): 피드백 값.
        """
        self.feedback = feedback_value
        self.receptor_sensitivity += 0.01 * feedback_value
        self.receptor_sensitivity = max(0.1, min(2.0, self.receptor_sensitivity))  # 민감도 제한

    def update_weight(self, pre_neuron_active, post_neuron_active, learning_rate=0.01):
        """
        Hebbian Learning 기반으로 시냅스 가중치 업데이트.

        Args:
            pre_neuron_active (bool): 프리뉴런의 활성 상태.
            post_neuron_active (bool): 포스트뉴런의 활성 상태.
            learning_rate (float): 학습률.
        """
        if pre_neuron_active and post_neuron_active:
            self.weight += learning_rate  # 활성화된 연결 강화
        else:
            self.weight *= 0.99  # 비활성화된 연결 약화
