import pytest
import streamlit as st
from streamlit.testing import TestSessionState
from unittest.mock import patch
from main import main

def test_initial_session_state():
    session = TestSessionState()
    main()
    assert 'messages' in session, "Session state should initialize with 'messages'."
    assert session.messages == [], "Messages should be initialized as an empty list."

def test_model_selection():
    session = TestSessionState()
    with patch("streamlit.sidebar.selectbox") as mock_selectbox:
        mock_selectbox.return_value = "phi3"
        main()
        assert mock_selectbox.called_once_with("Choose a model", ["llama3", "phi3", "mistral"], index=0)

def test_user_input():
    session = TestSessionState()
    with patch("streamlit.chat_input") as mock_chat_input:
        mock_chat_input.return_value = "Hello, AI!"
        main()
        assert len(session.messages) == 1, "There should be one message in session after user input."
        assert session.messages[0] == {"role": "user", "content": "Hello, AI!"}, "Message should be correctly added to session state."

def test_response_generation():
    session = TestSessionState()
    session.messages = [{"role": "user", "content": "Hello, AI!"}]

    with patch("streamlit.chat_input") as mock_chat_input, \
         patch("streamlit.chat_message"), \
         patch("streamlit.spinner"), \
         patch("utils.do_chat") as mock_do_chat:

        mock_chat_input.return_value = "Hello, AI!"
        main()

        assert mock_do_chat.called_once, "The do_chat function should be called once after user input."
        assert session.messages[-1]["role"] == "assistant", "The last message should be from the assistant."

def test_error_handling():
    session = TestSessionState()
    session.messages = [{"role": "user", "content": "Hello, AI!"}]

    with patch("streamlit.chat_input") as mock_chat_input, \
         patch("streamlit.chat_message"), \
         patch("streamlit.spinner"), \
         patch("utils.do_chat", side_effect=Exception("Test Error")), \
         patch("streamlit.error") as mock_error:

        mock_chat_input.return_value = "Hello, AI!"
        main()

        assert mock_error.called_once_with("An error occurred while generating the response.")
        assert session.messages[-1]["content"] == "Test Error", "The error message should be stored in session state."
