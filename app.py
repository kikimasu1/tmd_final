import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime
import re
from itertools import chain
import time
import base64
import io
import nltk

# ---- make sure NLTK "stopwords" corpus is present ----
try:
    from nltk.corpus import stopwords
    _ = stopwords.words("english")          # quick test-load
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
# Set page configuration
st.set_page_config(
    page_title="Bluesky Topic & Sentiment Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1976D2;
        margin-bottom: 0.5rem;
    }
    .st-emotion-cache-1qfvqdf {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<h1 class='main-header'>📊 Bluesky Topic & Sentiment Explorer</h1>", unsafe_allow_html=True)
st.markdown("""
This dashboard analyzes Bluesky posts using advanced NLP techniques to identify topics, 
sentiment patterns, and engagement trends. Use the filters to explore specific segments of the data.
""")

# ---------- Data loading with progress indicator ----------
@st.cache_data(show_spinner=False)
def load_data():
    with st.spinner('Loading and processing data...'):
        df = pd.read_csv("bluesky_posts_with_topics.csv")
        
        # ---- Process hashtags ----
        if "hashtags" in df.columns:
            if df["hashtags"].notna().any() and isinstance(df["hashtags"].dropna().iloc[0], str):
                # Convert stringified lists -> list
                df["hashtags"] = df["hashtags"].apply(
                    lambda x: eval(x) if isinstance(x, str) else ([] if pd.isna(x) else x)
                )
        else:
            tag_regex = re.compile(r'#\\w+')
            df["hashtags"] = df["text"].fillna('').str.lower().apply(lambda x: tag_regex.findall(x))
        
        # ---- Add/process date fields ----
        if "date" not in df.columns:
            if "createdAt" in df.columns:
                df["date"] = pd.to_datetime(df["createdAt"], errors="coerce").dt.date
            else:
                df["date"] = pd.to_datetime("today").date()
        else:
            # Ensure date is in datetime format
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        
        # ---- Process sentiment if needed ----
        if "sentiment" in df.columns:
            df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
        else:
            df["sentiment"] = 0.0
            
        return df

try:
    with st.spinner('Loading data...'):
        df = load_data()
    st.success('Data loaded successfully!', icon="✅")
    time.sleep(0.5)  # Short pause to show success message
    st.empty()  # Clear success message
except Exception as e:
    st.error(f"❌ Failed to load data: {str(e)}")
    st.stop()

# ------------ Sidebar filters -------------
st.sidebar.markdown("<h2>🔍 Filters</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Use these filters to explore specific segments of the data.")

# Topic selector with counts
topic_counts = df["topic_label"].value_counts()
topic_options = ["All"] + [f"{topic} ({count})" for topic, count in topic_counts.items()]
selected_topic_with_count = st.sidebar.selectbox("🎯 Filter by topic:", topic_options)

# Extract just the topic name without the count
if selected_topic_with_count != "All":
    selected_topic = selected_topic_with_count.split(" (")[0]
else:
    selected_topic = "All"

# Date range with min/max display
min_date = pd.to_datetime(df["date"]).min().date()
max_date = pd.to_datetime(df["date"]).max().date()
st.sidebar.markdown(f"📅 Date range (from {min_date} to {max_date})")
date_range = st.sidebar.date_input(
    "Select date range:",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Sentiment range slider
min_sentiment = float(df["sentiment"].min())
max_sentiment = float(df["sentiment"].max())
sentiment_range = st.sidebar.slider(
    "😊 Filter by sentiment score:", 
    min_sentiment, 
    max_sentiment, 
    (min_sentiment, max_sentiment), 
    step=0.1
)

# Apply base filters
df_filtered = df.copy()

# Date filter
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (pd.to_datetime(df_filtered["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(df_filtered["date"]) <= pd.to_datetime(end_date))
    df_filtered = df_filtered[mask]

# Topic filter
if selected_topic != "All":
    df_filtered = df_filtered[df_filtered["topic_label"] == selected_topic]

# Sentiment filter
df_filtered = df_filtered[df_filtered["sentiment"].between(sentiment_range[0], sentiment_range[1])]



# Hashtag selector (after topic/date/sentiment filtering)
all_hashtags = list(chain.from_iterable(df_filtered["hashtags"].dropna()))
distinct_hashtags = ["All"] + [tag for tag, count in Counter(all_hashtags).most_common(30)]
selected_hashtag = st.sidebar.selectbox("🔖 Filter by hashtag:", distinct_hashtags)

if selected_hashtag != "All":
    df_filtered = df_filtered[df_filtered["hashtags"].apply(lambda tags: selected_hashtag in tags if isinstance(tags, list) else False)]

# Additional options section
st.sidebar.markdown("---")
st.sidebar.markdown("<h3>Additional Options</h3>", unsafe_allow_html=True)

# Export filtered data option
if st.sidebar.button("📤 Export Filtered Data"):
    csv_buffer = io.StringIO()
    df_filtered.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    b64 = base64.b64encode(csv_str.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="bluesky_filtered_data.csv">Click to download CSV file</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

# Dark/light mode toggle
theme_mode = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])
if theme_mode == "Dark":
    plt.style.use("dark_background")
    chart_template = "plotly_dark"
    color_scheme = "Plasma"
    bg_color = "black"
else:
    plt.style.use("default")
    chart_template = "plotly_white"
    color_scheme = "Viridis"
    bg_color = "white"

# ----------- Header metrics --------------
if len(df_filtered) == 0:
    st.warning("⚠️ No posts match your filter criteria. Please adjust your filters.")
    st.stop()

st.markdown("<h2 class='sub-header'>📈 Key Metrics</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Posts", f"{len(df_filtered):,}", delta=f"{len(df_filtered)/len(df)*100:.1f}% of total")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.metric("Topics", df_filtered["topic_label"].nunique())
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    avg_sentiment = round(df_filtered["sentiment"].mean(), 3)
    sentiment_emoji = "😊" if avg_sentiment > 0.2 else "😐" if avg_sentiment > -0.2 else "☹️"
    st.metric("Avg. Sentiment", f"{avg_sentiment} {sentiment_emoji}", 
             delta=f"{avg_sentiment - df['sentiment'].mean():.3f} vs overall")
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    hash_count = len(set(all_hashtags))
    st.metric("Distinct Hashtags", f"{hash_count:,}")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------- Tabs --------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Topic Map", 
    "☁️ Word Cloud", 
    "📈 Sentiment Trend", 
    "🔍 Topic Breakdown",
    "📝 Post Explorer"
])

with tab1:
    st.markdown("<h3 class='sub-header'>Topic Map (t-SNE Visualization)</h3>", unsafe_allow_html=True)
    st.markdown("""
    This visualization shows how posts cluster into topics in a 2D space. 
    Each point represents a post, with color indicating the assigned topic.
    Proximity between points suggests content similarity.
    """)
    
    if {'x','y'}.issubset(df_filtered.columns):
        # Add hover data based on available columns
        hover_data = ['text', 'sentiment_label']
        if 'author' in df_filtered.columns:
            hover_data.append('author')
        if 'date' in df_filtered.columns:
            hover_data.append('date')
            
        # Create figure with improved styling
        fig = px.scatter(
            df_filtered,
            x='x', y='y',
            color='topic_label',
            hover_data=hover_data,
            height=600,
            template=chart_template,
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.75
        )
        
        # Enhance figure layout
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(
            legend_title_text='Topics',
            xaxis_title="t-SNE dimension 1",
            yaxis_title="t-SNE dimension 2",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("t‑SNE coordinates were not found in the dataset.")

# ---------- Word-Cloud (TF-IDF weighted) ----------
with tab2:
    st.markdown("<h3 class='sub-header'>Word Cloud (TF-IDF-weighted)</h3>", unsafe_allow_html=True)
    st.markdown("The cloud emphasises terms that are both frequent **and** unusually specific to the selected posts.")

    # combine text
    corpus = df_filtered["text"].dropna().astype(str).tolist()
    if not corpus:
        st.info("No text data available for word-cloud generation.");  st.stop()

    import nltk, string, emoji
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer

    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

    EN_STOP = set(stopwords.words("english"))
    BS_STOP = {"bluesky", "bsky", "post", "posts", "just", "like", "get", "got", "really",
               "time", "day", "today", "now", "see", "think", "new"}   # tweak as you wish
    STOPWORDS = EN_STOP | BS_STOP

    lemmatizer = WordNetLemmatizer()
    tok_re = re.compile(r"[a-z]{3,}")

    def norm(text: str) -> str:
        """lower-case, strip URLs/emojis/punctuation, lemmatise, drop stop-words."""
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(r"https?://\\S+", "", text.lower())                # URLs
        text = re.sub(r"@[\\w_]+", "", text)                             # mentions
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = tok_re.findall(text)
        lemmas = [lemmatizer.lemmatize(w) for w in words if w not in STOPWORDS]
        return " ".join(lemmas)

    docs = [norm(t) for t in corpus]

    # TF-IDF weighting – treat hashtags as separate tokens with double weight
    def tokenize(text):
        tags = [h[1:] for h in re.findall(r"#(\\w+)", text)]
        base = text.split()
        return base + tags  # tags duplicated → double weight

    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, min_df=2, max_df=0.7)
    tfidf = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()
    weights = tfidf.sum(axis=0).A1
    freq_dict = {t: w for t, w in zip(terms, weights) if t not in STOPWORDS and len(t) > 2}

    if not freq_dict:
        st.info("Nothing after cleaning – try relaxing filters.");  st.stop()

    # --- show alongside top list ---------------------------------
    col_wc, col_bar = st.columns([3, 1])
    with col_wc:
        wc = WordCloud(width=800, height=400,
                       background_color=bg_color,
                       colormap=color_scheme.lower(),
                       max_words=150).generate_from_frequencies(freq_dict)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    with col_bar:
        top_df = (pd.Series(freq_dict)
                    .sort_values(ascending=False)
                    .head(25)
                    .reset_index()
                    .rename(columns={"index": "Term", 0: "Weight"}))
        st.dataframe(top_df, height=450)

with tab3:
    st.markdown("<h3 class='sub-header'>Sentiment & Volume Over Time</h3>", unsafe_allow_html=True)
    st.markdown("""
    This chart tracks post volume and average sentiment over time.
    The line shows sentiment trends while bars represent post count.
    """)
    
    if 'date' in df_filtered.columns and df_filtered['date'].notna().any():
        df_filtered['date_dt'] = pd.to_datetime(df_filtered['date'])
        
        # Determine appropriate time grouping based on date range
        date_range_days = (df_filtered['date_dt'].max() - df_filtered['date_dt'].min()).days
        
        if date_range_days > 180:  # If more than 6 months, group by month
            time_grouper = df_filtered['date_dt'].dt.to_period('M').dt.to_timestamp()
            time_format = '%b %Y'
        elif date_range_days > 30:  # If more than a month, group by week
            time_grouper = df_filtered['date_dt'].dt.to_period('W').dt.to_timestamp()
            time_format = 'Week %U, %Y'
        else:  # Otherwise group by day
            time_grouper = df_filtered['date_dt'].dt.date
            time_format = '%b %d, %Y'
            
        # Group by the determined time period
        ts = df_filtered.groupby(time_grouper).agg(
            avg_sent=('sentiment','mean'), 
            posts=('text','count'),
            pos_sent=('sentiment', lambda x: (x > 0.2).sum()),
            neg_sent=('sentiment', lambda x: (x < -0.2).sum()),
            neu_sent=('sentiment', lambda x: ((x >= -0.2) & (x <= 0.2)).sum())
        ).reset_index()
        
        # Format time for display
        ts['formatted_date'] = ts['date_dt'].dt.strftime(time_format)
        
        if len(ts) > 1:
            # Create tabs for different time series visualizations
            ts_tab1, ts_tab2 = st.tabs(["Sentiment & Volume", "Sentiment Distribution"])
            
            with ts_tab1:
                fig = go.Figure()
                
                # Add sentiment line
                fig.add_trace(go.Scatter(
                    x=ts['date_dt'], 
                    y=ts['avg_sent'], 
                    name='Avg Sentiment', 
                    yaxis='y', 
                    line=dict(width=3, color='#1E88E5')
                ))
                
                # Add volume bars
                fig.add_trace(go.Bar(
                    x=ts['date_dt'], 
                    y=ts['posts'], 
                    name='Post Count', 
                    yaxis='y2', 
                    opacity=0.4,
                    marker_color='#26A69A'
                ))
                
                # Layout with dual y-axes
                fig.update_layout(
                    template=chart_template,
                    yaxis=dict(
                        title='Avg Sentiment', 
                        range=[-1,1],
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=['-1.0<br>Very Negative', '-0.5<br>Negative', '0.0<br>Neutral', '0.5<br>Positive', '1.0<br>Very Positive'],
                        gridcolor='lightgrey' if theme_mode == "Light" else 'darkgrey'
                    ),
                    yaxis2=dict(
                        title='Post Count', 
                        overlaying='y', 
                        side='right',
                        gridcolor='lightgrey' if theme_mode == "Light" else 'darkgrey'
                    ),
                    xaxis=dict(
                        title='Date',
                        gridcolor='lightgrey' if theme_mode == "Light" else 'darkgrey'
                    ),
                    legend=dict(orientation='h', y=-0.2),
                    height=500,
                    margin=dict(t=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with ts_tab2:
                # Create stacked area chart of sentiment distribution
                ts['positive_pct'] = ts['pos_sent'] / ts['posts'] * 100
                ts['neutral_pct'] = ts['neu_sent'] / ts['posts'] * 100
                ts['negative_pct'] = ts['neg_sent'] / ts['posts'] * 100
                
                fig = go.Figure()
                
                # Add traces for sentiment percentages
                fig.add_trace(go.Scatter(
                    x=ts['date_dt'], y=ts['positive_pct'],
                    mode='lines',
                    line=dict(width=0),
                    stackgroup='one',
                    name='Positive',
                    fillcolor='rgba(76, 175, 80, 0.8)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=ts['date_dt'], y=ts['neutral_pct'],
                    mode='lines',
                    line=dict(width=0),
                    stackgroup='one',
                    name='Neutral',
                    fillcolor='rgba(158, 158, 158, 0.8)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=ts['date_dt'], y=ts['negative_pct'],
                    mode='lines',
                    line=dict(width=0),
                    stackgroup='one',
                    name='Negative',
                    fillcolor='rgba(239, 83, 80, 0.8)'
                ))
                
                fig.update_layout(
                    template=chart_template,
                    yaxis=dict(
                        title='Percentage of Posts',
                        ticksuffix='%',
                        range=[0, 100]
                    ),
                    xaxis_title='Date',
                    legend=dict(orientation='h', y=-0.2),
                    height=500,
                    margin=dict(t=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least two dates to plot a trend.")
    else:
        st.info("Date information is not available for trend analysis.")

with tab4:
    st.markdown("<h3 class='sub-header'>Topic Breakdown Analysis</h3>", unsafe_allow_html=True)
    
    # Topic distribution pie chart
    topic_counts = df_filtered['topic_label'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    
    # Calculate percentages
    topic_counts['Percentage'] = topic_counts['Count'] / topic_counts['Count'].sum() * 100
    
    # Topic sentiment analysis
    topic_sentiment = df_filtered.groupby('topic_label')['sentiment'].agg(['mean', 'count']).reset_index()
    topic_sentiment.columns = ['Topic', 'Avg Sentiment', 'Post Count']
    
    # Create columns for the visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Topic Distribution")
        fig = px.pie(
            topic_counts, 
            values='Count', 
            names='Topic',
            template=chart_template,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Topic Sentiment Comparison")
        # Sort by sentiment for better visualization
        topic_sentiment = topic_sentiment.sort_values('Avg Sentiment')
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars with custom styling based on sentiment
        colors = ['#ef5350' if x < -0.2 else '#9e9e9e' if x < 0.2 else '#66bb6a' for x in topic_sentiment['Avg Sentiment']]
        
        fig.add_trace(go.Bar(
            x=topic_sentiment['Avg Sentiment'],
            y=topic_sentiment['Topic'],
            orientation='h',
            marker_color=colors,
            text=topic_sentiment['Post Count'].apply(lambda x: f"{x:,} posts"),
            textposition='auto'
        ))
        
        # Add vertical line at 0
        fig.add_shape(
            type="line",
            x0=0, y0=-0.5,
            x1=0, y1=len(topic_sentiment)-0.5,
            line=dict(color="gray", width=1, dash="dot")
        )
        
        # Update layout
        fig.update_layout(
            template=chart_template,
            xaxis_title="Average Sentiment Score",
            xaxis=dict(range=[-1, 1]),
            margin=dict(t=0, b=0, l=0, r=0),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Topic engagement analysis (if engagement metrics available)
    if all(col in df_filtered.columns for col in ['replyCount', 'repostCount', 'likeCount']):
        st.markdown("#### Topic Engagement Analysis")
        
        # Calculate engagement metrics by topic
        topic_engagement = df_filtered.groupby('topic_label').agg(
            posts=('text', 'count'),
            replies=('replyCount', 'sum'),
            reposts=('repostCount', 'sum'),
            likes=('likeCount', 'sum')
        ).reset_index()
        
        # Calculate per-post metrics
        topic_engagement['replies_per_post'] = topic_engagement['replies'] / topic_engagement['posts']
        topic_engagement['reposts_per_post'] = topic_engagement['reposts'] / topic_engagement['posts']
        topic_engagement['likes_per_post'] = topic_engagement['likes'] / topic_engagement['posts']
        
        # Melt the dataframe for easier plotting
        engagement_melt = pd.melt(
            topic_engagement,
            id_vars=['topic_label'],
            value_vars=['replies_per_post', 'reposts_per_post', 'likes_per_post'],
            var_name='Metric',
            value_name='Value'
        )
        
        # Clean up metric names for display
        engagement_melt['Metric'] = engagement_melt['Metric'].map({
            'replies_per_post': 'Avg Replies',
            'reposts_per_post': 'Avg Reposts',
            'likes_per_post': 'Avg Likes'
        })
        
        # Create grouped bar chart
        fig = px.bar(
            engagement_melt,
            x='topic_label',
            y='Value',
            color='Metric',
            barmode='group',
            template=chart_template,
            labels={'topic_label': 'Topic', 'Value': 'Average Count Per Post'},
            color_discrete_sequence=['#42A5F5', '#66BB6A', '#FFA726']
        )
        
        fig.update_layout(
            xaxis={'categoryorder':'total descending'},
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("<h3 class='sub-header'>Post Explorer</h3>", unsafe_allow_html=True)
    
    # Search box for filtering posts
    search_term = st.text_input("🔍 Search posts for specific content:")
    
    df_display = df_filtered.copy()
    
    if search_term:
        # Filter dataframe based on search term
        df_display = df_display[df_display['text'].str.contains(search_term, case=False, na=False)]
        st.write(f"Found {len(df_display)} posts containing '{search_term}'")
    
    # Column selection
    display_cols = ['date', 'topic_label', 'sentiment_label', 'text']
    
    # Add author if available
    if 'author' in df_display.columns:
        display_cols.insert(1, 'author')
    
    # Add engagement columns if available
    engagement_cols = [col for col in ['likeCount', 'repostCount', 'replyCount'] if col in df_display.columns]
    if engagement_cols:
        display_cols.extend(engagement_cols)
    
    # Display options
    st.markdown("**View Options:**")
    display_option = st.radio(
        "Choose display format:",
        ["Table View", "Card View"],
        horizontal=True
    )
    
    # Number of posts to display
    num_posts = st.slider("Number of posts to display:", 10, 100, 25, 5)
    
    if display_option == "Table View":
        # Display as interactive table
        st.dataframe(
            df_display[display_cols].sort_values('date', ascending=False).head(num_posts),
            use_container_width=True,
            height=400
        )
    else:
        # Display as cards
        posts_to_display = df_display[display_cols].sort_values('date', ascending=False).head(num_posts)
        
        for _, post in posts_to_display.iterrows():
            # Create a card for each post
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Post:** {post['text']}")
            
            with col2:
                # Format metadata with colored sentiment
                sentiment = post['sentiment_label']
                sentiment_color = "#4CAF50" if "Positive" in sentiment else "#F44336" if "Negative" in sentiment else "#9E9E9E"
                
                metadata = f"""
                <strong>Topic:</strong> {post['topic_label']}<br>
                <strong>Date:</strong> {post['date']}<br>
                <strong>Sentiment:</strong> <span style="color:{sentiment_color}">{sentiment}</span><br>
                """
                
                # Add author if available
                if 'author' in post.index:
                    metadata += f"<strong>Author:</strong> {post['author']}<br>"
                
                # Add engagement metrics if available
                for col in engagement_cols:
                    readable_col = col.replace('Count', '')
                    metadata += f"<strong>{readable_col}s:</strong> {post[col]}<br>"
                
                st.markdown(metadata, unsafe_allow_html=True)
            
            st.markdown("---")

# Add footer with info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
Bluesky Topic & Sentiment Explorer | Data updated: {last_update} | Total posts: {total_posts}
</div>
""".format(
    last_update=max_date,
    total_posts=f"{len(df):,}"
), unsafe_allow_html=True)