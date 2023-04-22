# gui.py
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, QScrollArea
from PyQt5.QtCore import Qt
from xml.etree.ElementTree import Element, SubElement, ElementTree
import os

class ChatGUI:
    def __init__(self, agi):
        self.agi = agi

    def run(self):
        app = QApplication([])
        main_window = QWidget()
        main_layout = QVBoxLayout()

        # Create text box for user input
        input_box = QTextEdit()
        input_box.setFixedHeight(50)
        input_box.setPlaceholderText("Type your message here...")

        # Create send button
        send_button = QPushButton("Send")
        send_button.setFixedHeight(50)

        # Create chat history window
        chat_history = QScrollArea()
        chat_history.setWidgetResizable(True)
        chat_history_content = QWidget()
        chat_history_layout = QVBoxLayout(chat_history_content)
        chat_history.setWidget(chat_history_content)

        # Load chat history from chat-history.xml file, if it exists
        if os.path.exists("chat-history.xml"):
            chat_history_root = ElementTree(file="chat-history.xml").getroot()
            for message in chat_history_root.findall("message"):
                message_box = QLabel(f"{message.find('user').text}: {message.find('bot').text}")
                message_box.setStyleSheet("background-color: #EEEEEE; padding: 10px; border-radius: 5px;")
                message_box.setAlignment(Qt.AlignRight if message.find('user').text == 'User' else Qt.AlignLeft)
                chat_history_layout.addWidget(message_box)

        # Add components to main layout
        main_layout.addWidget(chat_history)
        input_and_send_layout = QHBoxLayout()
        input_and_send_layout.addWidget(input_box)
        input_and_send_layout.addWidget(send_button)
        main_layout.addLayout(input_and_send_layout)

        main_window.setLayout(main_layout)

        # Connect send button to send_message method
        send_button.clicked.connect(lambda: self.send_message(input_box.toPlainText(), chat_history_layout))

        main_window.show()
        app.exec_()

    def send_message(self, message, chat_history_layout):
        user_message_box = QLabel(f"User: {message}")
        user_message_box.setStyleSheet("background-color: #EEEEEE; padding: 10px; border-radius: 5px;")
        user_message_box.setAlignment(Qt.AlignRight)
        chat_history_layout.addWidget(user_message_box)

        response = self.agi.generate_response(message)
        bot_message_box = QLabel(f"Bot: {response}")
        bot_message_box.setStyleSheet("background-color: #F0F0F0; padding: 10px; border-radius: 5px;")
        bot_message_box.setAlignment(Qt.AlignLeft)
        chat_history_layout.addWidget(bot_message_box)

        # Save chat history to chat-history.xml file
        chat_history_root = Element("chat-history")
        for i in range(chat_history_layout.count()):
            message = chat_history_layout.itemAt(i).widget()
            user_or_bot = message.text().split(": ")[0]
            text = message.text().split(": ")[1]
            message_element = Element("message")
            user_element = Element("user")
            user_element.text = user_or_bot
            message_element.append(user_element)
            bot_element = Element("bot")
            bot_element.text = text
            message_element.append(bot_element)
            chat_history_root.append(message_element)
        ElementTree(chat_history_root).write("chat-history.xml")