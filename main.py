from neuron_model import NeuronNetwork
from memory import MemoryManager
from retina_processor import RetinaProcessor
from cochlea_processor import CochleaProcessor
from text_processor import TextProcessor
import cupy as cp


def main():
    num_neurons = 1000000
    min_connections = 1000
    max_connections = 100000

    # 뉴런 네트워크 및 처리기 초기화
    network = NeuronNetwork(num_neurons, min_connections, max_connections)
    memory_manager = MemoryManager(network)
    retina_processor = RetinaProcessor(network)
    cochlea_processor = CochleaProcessor(network)
    text_processor = TextProcessor(network)

    print("Neuron-based AI with dynamic synapses and memory is ready. Type 'exit' to quit.")
    print("Commands: 'image', 'audio', 'text'")

    while True:
        user_input = input("Command: ")
        if user_input.lower() == "exit":
            break

        if user_input.lower() == "image":
            input_signals = retina_processor.capture_webcam()
        elif user_input.lower() == "audio":
            input_signals = cochlea_processor.capture_microphone()
        elif user_input.lower() == "text":
            text = input("Enter text: ")
            input_signals = text_processor.process_text(text)
        else:
            print("Invalid command.")
            continue

        # 뉴런 네트워크에 신호 전달 및 학습
        network.stimulate_neurons(cp.asarray(input_signals))
        network.update_neurons()
        network.update_weights()
        network.prune_and_rewire()

        # 활성화된 뉴런 가져오기
        active_neurons = network.get_active_neurons()

        # 기억 검색 및 출력
        recalled_memory = memory_manager.recall_memory(active_neurons)
        if recalled_memory:
            print(f"AI: I remember you said: {recalled_memory}")
        else:
            print("AI: This is new to me.")

        # 기억 저장 및 통합
        memory_manager.store_memory(user_input, active_neurons)
        memory_manager.consolidate_memory()


if __name__ == "__main__":
    main()
