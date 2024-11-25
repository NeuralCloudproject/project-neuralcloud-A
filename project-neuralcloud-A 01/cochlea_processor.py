import numpy as np


class TextProcessor:
    """
    텍스트 입력을 감각뉴런으로 변환하는 클래스.
    """

    def __init__(self, neuron_network):
        self.network = neuron_network

    def text_to_signals(self, text):
        """
        텍스트를 뉴런 신호로 변환.

        Args:
            text (str): 입력 텍스트.

        Returns:
            numpy array: 뉴런 입력 신호.
        """
        ascii_values = np.array([ord(char) for char in text])
        normalized = ascii_values / 255.0  # 정규화
        signals = np.pad(normalized, (0, self.network.num_neurons - len(normalized)), 'constant')
        return signals

    def process_text(self, text):
        """
        텍스트를 처리하여 뉴런 입력 신호로 변환.

        Args:
            text (str): 입력 텍스트.

        Returns:
            numpy array: 뉴런 입력 신호.
        """
        return self.text_to_signals(text)
