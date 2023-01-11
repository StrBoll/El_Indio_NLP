import streamlit as st
from streamlit_chat import message as st_message

from main import *

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Hello Chatbot")


def push_results():
    user_input = st.session_state.input_text
    user_result = generate_lemma(user_input)

    bow = bag(user_result)
    results = model.predict(numpy.array([bow]))[0]
    return_list = numpy.argmax(results)

    cat = classes[return_list]

    for tag in dictionary['intents']:
        if tag['tag'] == cat:
            responses = tag['responses']
            response = random.choice(responses)

            st.session_state.history.append({"message": user_input, "is_user": True})
            st.session_state.history.append({"message": response, "is_user": False})


st.text_input("Talk to the bot", key="input_text", on_change=push_results)

for chat in st.session_state.history:
    st_message(**chat)  # unpacking