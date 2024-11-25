from neuron_model import NeuronNetwork
from memory import MemoryManager
from retina_processor import RetinaProcessor
from cochlea_processor import CochleaProcessor
from text_processor import TextProcessor
from dialogue_manager import DialogueManager
import cupy as cp


def main():
    num_neurons = 15000000  # 전체 뉴런 수
    min_connections = 1000  # 최소 시냅스 연결 수
    max_connections = 100000  # 최대 시냅스 연결 수

    # 뉴런 네트워크 및 처리기 초기화
    network = NeuronNetwork(num_neurons, min_connections, max_connections)
    memory_manager = MemoryManager(network)
    dialogue_manager = DialogueManager(memory_manager)
    retina_processor = RetinaProcessor(network)
    cochlea_processor = CochleaProcessor(network)
    text_processor = TextProcessor(network)

    print("Neuron-based AI with sensory input and memory is ready.")
    print("Type 'exit' to quit. Available commands: 'image', 'audio', 'text'.")

    while True:
        user_input = input("Command: ")
        if user_input.lower() == "exit":
            break

        if user_input.lower() == "image":
            print("Processing image input...")
            input_signals = retina_processor.capture_webcam()
        elif user_input.lower() == "audio":
            print("Processing audio input...")
            input_signals = cochlea_processor.capture_microphone()
        elif user_input.lower() == "text":
            text = input("Enter text: ")
            input_signals = text_processor.process_text(text)
        else:
            print("Invalid command. Use 'image', 'audio', or 'text'.")
            continue

        print("Stimulating neurons...")
        network.stimulate_neurons(cp.asarray(input_signals))

        print("Updating neurons and synapses...")
        network.update_neurons()
        network.update_weights()

        active_neurons = network.get_active_neurons()
        print(f"Active neurons count: {len(active_neurons)}")

        response = dialogue_manager.respond(active_neurons)
        print(f"AI: {response}")

        print("Storing new memory...")
        memory_manager.store_memory(user_input, active_neurons)
        memory_manager.consolidate_memory()

    print("Exiting AI program. Goodbye!")


if __name__ == "__main__":
    main()
