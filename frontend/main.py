from typing import Set, Tuple

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message


st.header("Guvi BOT 🤖")

prompt = st.text_input(" ", placeholder="Enter your query : ")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_source_string(sources: Set[str]) -> str:
    if not sources:
        return ""
    sources_list = list(sources)
    sources_list.sort()
    sources_string = "sources\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with (st.spinner("Guvi Bot is trying hard to get you the best responce possible, Please wait 🕣.....")):
        generated_responce = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set(
            [doc.metadata["source"] for doc in generated_responce["source_documents"]]
        )

        formatted_responce = (
            f"{generated_responce['answer']}\n\n{create_source_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_responce)
        st.session_state["chat_history"].append((prompt, generated_responce['answer']))


if st.session_state["chat_answer_history"]:
    for user_prompt, generated_responce in zip(
        st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]
    ):
        message(user_prompt, is_user=True)
        message(generated_responce)
