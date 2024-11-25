import numpy as np
import sounddevice as sd


class CochleaProcessor:
    def __init__(self, neuron_network):
        self.network = neuron_network

    def process_audio(self, audio_data, sample_rate):
        """
        음성을 주파수 신호로 변환 (달팽이관 모방).
        """
        num_freq_bins = 128
        fft_result = np.abs(np.fft.rfft(audio_data))[:num_freq_bins]
        normalized = fft_result / np.max(fft_result)  # [0, 1]로 정규화
        signals = np.pad(normalized, (0, self.network.num_neurons - len(normalized)), 'constant')
        return signals

    def capture_microphone(self, duration=3):
        """
        마이크에서 음성을 캡처하고 달팽이관 처리.
        """
        audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
        sd.wait()
        return self.process_audio(audio_data.flatten(), 44100)
