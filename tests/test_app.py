import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Initialize the AppTest with your Streamlit app file
at = AppTest.from_file("pages/Basic_LLM.py")

def test_initialization():
    # Run the app to the initial state
    at.run()
    assert not at.exception, "The app should run without exceptions."
    assert len(at.session_state['messages']) == 0, "Session state 'messages' should be initialized as an empty list."

def test_model_selection():
    # Simulate selecting a model from the sidebar
    at.sidebar.selectbox[0].select("llama3").run()
    assert not at.exception, "Model selection should not cause any exceptions."

def test_user_input():
    # Simulate user input via the chat input
    at.chat_input[0].set_value("Hello, AI!").run()
    assert not at.exception, "User input should not cause any exceptions."
    assert at.session_state['messages'][-1]['content'] == "Hello, AI!", "The last message in session state should be the user input."

def test_response_generation():
    # Simulate generating a response from the assistant
    at.chat_input[0].set_value("Hello, AI!").run()
    assert not at.exception, "Response generation should not cause any exceptions."
    # Here, you'd check if the assistant's response is added to session state
    assert at.session_state['messages'][-1]['role'] == "assistant", "The last message should be from the assistant."

def test_do_chat_error_handling():
    with patch('utils.stream_chat', side_effect=Exception("Test Error")):
        at = AppTest.from_file("pages/Basic_LLM.py")

        # Simulate user input that will trigger the do_chat function
        at.chat_input[0].set_value("Hello, AI!").run()

        # Assert that the error was handled correctly and the message was appended to session state
        assert "Test Error" in at.session_state['messages'][-1]['content'], "The error message should be stored in session state."
