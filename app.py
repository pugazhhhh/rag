"""Streamlit chat UI for the AskAboutMe resume assistant."""
import streamlit as st

from query import retrieve, generate_answer

st.set_page_config(page_title="AskAboutMe", page_icon="\U0001F916", layout="centered")

st.markdown(
    """
    <style>
        #MainMenu, header, footer {visibility: hidden;}

        .brand-card {
            padding: 2rem 2.2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
            color: #ffffff;
            margin-bottom: 1.8rem;
        }
        .brand-card h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
        .brand-card p.tagline {
            margin: 0.4rem 0 0 0;
            font-size: 1.02rem;
            color: #93c5fd;
            font-weight: 500;
        }

        .stChatMessage {
            border-radius: 12px;
        }
    </style>

    <div class="brand-card">
        <h1>AskAboutMe</h1>
        <p class="tagline">Your guide to my professional journey</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Ask a question about my experience, skills, or projects to get started.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.button("Clear conversation", disabled=True, help="Nothing to clear yet")
else:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question, e.g. 'What tracking algorithms have you used?'")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧑‍💻"):
        with st.spinner("Thinking..."):
            chunks, _ = retrieve(question, n_results=4)
            reply = generate_answer(question, chunks)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
