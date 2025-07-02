import streamlit as st
import requests

st.set_page_config(
    page_title="AI Literature Review Agent",
    page_icon="📚",
    layout="centered",
)

st.title("📚 AI Literature Review Agent")
st.write("Enter a research topic and number of papers to generate a literature review.")

topic = st.text_input("Research topic", value="Autogen")
num_papers = st.number_input("Number of papers", min_value=1, max_value=10, value=2, step=1)

if st.button("Generate Literature Review"):
    with st.spinner("Generating your literature review..."):
        try:
            response = requests.post(
                "http://localhost:5000/api/literature_review",
                json={"topic": topic, "num_papers": num_papers},
                timeout=600
            )
            data = response.json()
            summary = data.get('content', '')
            if summary:
                st.markdown(summary, unsafe_allow_html=True)
            elif 'error' in data:
                st.error(f"Error: {data['error']}")
            else:
                st.warning("No summary returned. Please try again.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
else:
    st.info("Enter a topic and number of papers, then click 'Generate Literature Review'.")
