from retina_processor import RetinaProcessor
from cochlea_processor import CochleaProcessor
from text_processor import TextProcessor


class SensoryNeuron:
    """
    감각 데이터를 받아 연합뉴런으로 전달하는 감각뉴런 통합 클래스.
    """

    def __init__(self, neuron_network):
        """
        감각뉴런 초기화.

        Args:
            neuron_network (NeuronNetwork): 연결된 연합뉴런 네트워크.
        """
        self.neuron_network = neuron_network
        # 개별 감각 처리 클래스 초기화
        self.visual_processor = RetinaProcessor(neuron_network)  # 시각
        self.audio_processor = CochleaProcessor(neuron_network)  # 청각
        self.text_processor = TextProcessor(neuron_network)  # 텍스트

    def process_visual_input(self):
        """
        카메라 입력을 처리하여 뉴런 네트워크로 전달.
        """
        print("Processing visual input...")
        signals = self.visual_processor.capture_webcam()
        self.neuron_network.stimulate_neurons(signals)

    def process_audio_input(self, duration=3):
        """
        마이크 입력을 처리하여 뉴런 네트워크로 전달.

        Args:
            duration (int): 마이크로 캡처할 오디오 길이 (초 단위).
        """
        print("Processing audio input...")
        signals = self.audio_processor.capture_microphone(duration)
        self.neuron_network.stimulate_neurons(signals)

    def process_text_input(self, text):
        """
        사용자 텍스트 입력을 처리하여 뉴런 네트워크로 전달.

        Args:
            text (str): 사용자 입력 텍스트.
        """
        print("Processing text input...")
        signals = self.text_processor.process_text(text)
        self.neuron_network.stimulate_neurons(signals)
