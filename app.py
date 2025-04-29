import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 Interactive Bluesky Topic Explorer")

# Load processed CSV
df = pd.read_csv("bluesky_posts_with_topics.csv")

# Filter options
selected_topic = st.selectbox("🎯 Filter by topic:", ["All"] + sorted(df["topic_label"].unique()))
sentiment_range = st.slider("😊 Filter by sentiment score", -1.0, 1.0, (-1.0, 1.0))

# Apply filters
filtered = df[df["sentiment"].between(sentiment_range[0], sentiment_range[1])]
if selected_topic != "All":
    filtered = filtered[filtered["topic_label"] == selected_topic]

# Plotly scatter plot (interactive)
fig = px.scatter(
    filtered,
    x="x", y="y",
    size="size", color="topic_label",
    hover_data={"text": True, "sentiment": True, "x": False, "y": False},
    labels={"x": "t-SNE X", "y": "t-SNE Y"},
    title="🧠 Neural Topic Map (Interactive)"
)
# fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')))

st.plotly_chart(fig, use_container_width=True)

# Optional: raw data table
with st.expander("🗃 Show data table"):
    st.dataframe(filtered[["text", "topic_label", "sentiment"]])