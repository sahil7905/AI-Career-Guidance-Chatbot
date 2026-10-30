
"""AI-Powered Career Guidance Chatbot - Streamlit UI.
Run: streamlit run app.py
"""

import re
import html
import streamlit as st

from src.llm import generate, llm_available
from src.rag import retrieve


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Career Guidance Chatbot",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 AI-Powered Career Guidance Chatbot")
st.caption(
    "RAG + LLM: Personalised career paths, skills and resources "
    "based on your profile."
)


# --------------------------------------------------
# PROFILE OPTIONS
# --------------------------------------------------

SKILLS = [
    "Python",
    "Java",
    "C/C++",
    "SQL",
    "JavaScript",
    "HTML/CSS",
    "Machine Learning",
    "Data Analysis",
    "Cloud (AWS/Azure/GCP)",
    "Linux",
    "Networking",
    "Communication",
    "Problem Solving",
    "Excel",
]

INTERESTS = [
    "Artificial Intelligence",
    "Web Development",
    "Data Science",
    "Cloud & DevOps",
    "Cybersecurity",
    "Mobile Apps",
    "Product & Management",
    "Research",
]


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []


# --------------------------------------------------
# RESPONSE CLEANING
# --------------------------------------------------

def clean_ai_response(text: str) -> str:
    """Clean common HTML formatting artifacts in model responses."""

    if not isinstance(text, str):
        return "Sorry, no response was generated."

    text = html.unescape(text)

    # Convert HTML line breaks to actual new lines.
    text = re.sub(
        r"<br\s*/?>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    # Convert basic paragraph boundaries.
    text = re.sub(
        r"</p\s*>",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"<p(?:\s+[^>]*)?>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove accidental HTML tags commonly emitted by models.
    text = re.sub(
        r"</?(?:div|span)\b[^>]*>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# SIDEBAR: USER PROFILE
# --------------------------------------------------

with st.sidebar:
    st.header("👤 Your Profile")

    name = st.text_input(
        "Name",
        placeholder="Enter your name",
    )

    education = st.selectbox(
        "Education",
        [
            "B.Tech / B.E.",
            "BCA / B.Sc",
            "MCA / M.Tech",
            "Diploma",
            "Other",
        ],
    )

    level = st.selectbox(
        "Experience",
        [
            "Student / Fresher",
            "0-2 years",
            "2-5 years",
            "5+ years",
        ],
    )

    skills = st.multiselect(
        "Current skills",
        SKILLS,
    )

    interests = st.multiselect(
        "Interests",
        INTERESTS,
    )

    goal = st.text_area(
        "Career goal (optional)",
        placeholder="e.g. Become an ML engineer at a product company",
    )

    generate_clicked = st.button(
        "🚀 Generate Career Guidance",
        use_container_width=True,
        type="primary",
    )

    clear_clicked = st.button(
        "🧹 Clear chat",
        use_container_width=True,
    )

    if clear_clicked:
        st.session_state.chat = []
        st.session_state.last_contexts = []
        st.rerun()


# --------------------------------------------------
# BUILD USER PROFILE
# --------------------------------------------------

profile = (
    f"Name: {name or 'N/A'}; "
    f"Education: {education}; "
    f"Experience: {level}; "
    f"Skills: {', '.join(skills) or 'none listed'}; "
    f"Interests: {', '.join(interests) or 'none listed'}; "
    f"Goal: {goal or 'not specified'}"
)


# --------------------------------------------------
# API STATUS
# --------------------------------------------------

if not llm_available():
    st.warning(
        "GROQ_API_KEY not found. Offline mode is active; "
        "only retrieved knowledge will be displayed."
    )


# --------------------------------------------------
# ANSWER GENERATION
# --------------------------------------------------

def answer(question: str):
    """Retrieve relevant context and generate a grounded response."""

    # Retrieve fewer chunks to reduce prompt size.
    contexts = retrieve(
        f"{profile}. {question}",
        k=3,
    )

    if llm_available():
        # Send only recent conversation messages.
        # The current question is added separately below.
        history = [
            {
                "role": message["role"],
                "content": message["content"][:1200],
            }
            for message in st.session_state.chat[-4:]
            if message.get("role") in ("user", "assistant")
            and isinstance(message.get("content"), str)
        ]

        response = generate(
            profile,
            question,
            contexts,
            history,
        )

        response = clean_ai_response(response)

    else:
        response_parts = [
            f"**{c.get('source', 'Knowledge base')}**\n\n"
            f"{c.get('text', '')}"
            for c in contexts
        ]

        response = (
            "**Offline mode — relevant knowledge found:**\n\n"
            + "\n\n".join(response_parts)
        )

    return response, contexts


# --------------------------------------------------
# DISPLAY PREVIOUS CHAT
# --------------------------------------------------

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(
            clean_ai_response(message["content"])
        )


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

question = None

if generate_clicked:
    question = (
        "Based on my profile, suggest suitable career paths, "
        "analyse my skill gaps, create a 30-60-90 day learning "
        "roadmap, and recommend relevant resources."
    )

user_question = st.chat_input(
    "Ask a follow-up (e.g. 'How do I become a data scientist?')"
)

if user_question and user_question.strip():
    question = user_question.strip()


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:
    # Keep the current question in the history before calling answer().
    st.session_state.chat.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching knowledge base and generating guidance..."
            ):
                response, contexts = answer(question)

            st.markdown(response)

            st.session_state.last_contexts = contexts

            # Show retrieved sources and context.
            with st.expander("📚 Retrieved context (RAG)"):
                if contexts:
                    for context in contexts:
                        source = context.get(
                            "source",
                            "Knowledge base",
                        )
                        distance = context.get("distance")

                        if isinstance(distance, (int, float)):
                            st.markdown(
                                f"**{source}** "
                                f"(distance: {distance:.3f})"
                            )
                        else:
                            st.markdown(f"**{source}**")

                        st.write(context.get("text", ""))

                        st.divider()
                else:
                    st.info("No relevant context was retrieved.")

        except Exception as error:
            response = (
                "Sorry, an error occurred while generating "
                "career guidance.\n\n"
                "Please try a shorter question or check your "
                "Groq API configuration and terminal logs."
            )

            st.error(response)

            # Keep technical details in the terminal, not in the UI.
            print(f"Career guidance error: {error}")

        # Store the assistant response for subsequent turns.
        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
