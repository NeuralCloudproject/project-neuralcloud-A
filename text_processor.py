import numpy as np


class TextProcessor:
    def __init__(self, neuron_network):
        self.network = neuron_network

    def text_to_signals(self, text):
        """
        텍스트를 뉴런 신호로 변환.
        """
        ascii_values = np.array([ord(char) for char in text])
        normalized = ascii_values / 255.0
        signals = np.pad(normalized, (0, self.network.num_neurons - len(normalized)), 'constant')
        return signals

    def process_text(self, text):
        """
        입력 텍스트를 처리하여 뉴런 네트워크에 전달할 신호로 변환.
        """
        return self.text_to_signals(text)
