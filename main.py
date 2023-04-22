# main.py
from agi import *
from agi import AGI
from gui import ChatGUI
import sys

def mode_1():
    # Initialize the AGI
    agi = AGI("gpt4all-ggml.bin", "emotional_decoder.csv")

    # Read input from external program
    while True:
        user_input = input()
        response = agi.generate_response(user_input)
        print(response)

def mode_2():
    # Initialize the AGI
    agi = AGI("gpt4all-ggml.bin", "emotional_decoder.csv")

    # Start GUI
    chat_gui = ChatGUI(agi)
    chat_gui.run()

def main():
    # Check for command-line arguments to determine which mode to run in
    mode = 2

    if mode == 1:
        mode_1()
    elif mode == 2:
        # Initialize the AGI
        agi = AGI("gpt4all-ggml.bin", "emotional_decoder.csv")

        # Start GUI
        chat_gui = ChatGUI(agi)
        chat_gui.run()

if __name__ == "__main__":
    AGI("gpt4all-ggml.bin", "emotional_decoder.csv")
    main()