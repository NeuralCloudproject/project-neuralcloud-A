class DialogueManager:
    """
    사용자와 대화를 관리하는 클래스.
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    def respond(self, active_neurons):
        """
        활성화된 뉴런을 기반으로 적절한 응답 생성.

        Args:
            active_neurons (numpy array): 활성화된 뉴런의 인덱스.

        Returns:
            str: 생성된 응답.
        """
        remembered = self.memory_manager.recall_memory(active_neurons)
        if remembered:
            return f"I remember: {remembered}"
        else:
            return "This is new to me. What else can you tell me?"
