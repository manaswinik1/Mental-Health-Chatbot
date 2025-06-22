"""Streamlit interface for the Mental Health Chatbot."""

import streamlit as st

from src.data_loader import load_data
from src.response_generator import generate_response
from src.safety_filter import filter_response


def main() -> None:
    """Run the Streamlit app."""
    st.title("Mental Health Chatbot")
    st.write(
        "This chatbot provides empathetic responses based on a preloaded dataset."
    )

    user_input = st.text_input("How are you feeling today?")
    if st.button("Submit") and user_input:
        data = load_data()
        raw_response = generate_response(user_input, data)
        safe_response = filter_response(raw_response)
        st.markdown(f"**Chatbot:** {safe_response}")


if __name__ == "__main__":
    main()
