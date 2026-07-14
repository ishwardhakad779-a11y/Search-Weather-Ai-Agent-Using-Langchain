import os
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# ----------------------------
# Tools & Agent (unchanged logic)
# ----------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city name."""
    api_key = os.environ["OPENWEATHER_API_KEY"]
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        return f"Sorry, couldn't fetch weather for {city}. Error: {data.get('message', 'unknown error')}"
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    description = data["weather"][0]["description"]
    humidity = data["main"]["humidity"]
    return f"Weather in {city}: {temp}°C, feels like {feels_like}°C, {description}, humidity {humidity}%"


@st.cache_resource
def get_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, api_key=GROQ_API_KEY)
    search_tool = TavilySearchResults()
    tools = [search_tool, get_weather]
    agent = create_react_agent(
        llm,
        tools,
        prompt="You are a helpful assistant with access to tools. Use them when needed to answer accurately.",
    )
    return agent


# ----------------------------
# Page config & styling
# ----------------------------
st.set_page_config(page_title="AI Agent Assistant", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 14px;
    }
    .title-text {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        text-align: center;
        color: #9aa0a6;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title-text">🤖 AI Agent Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Ask about stock prices, weather, or anything the web can answer</div>',
    unsafe_allow_html=True,
)

# ----------------------------
# Chat state
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("e.g. What is the AAPL stock price today? Find its current weather too.")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                response = agent.invoke({"messages": [("user", user_input)]})
                answer = response["messages"][-1].content
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"
            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))

with st.sidebar:
    st.header("ℹ️ About")
    st.write("This assistant can:")
    st.markdown("- 🔎 Search the web (Tavily)\n- ☁️ Fetch live weather (OpenWeather)\n- 📈 Answer general questions via Groq's Llama 3.3 70B")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()