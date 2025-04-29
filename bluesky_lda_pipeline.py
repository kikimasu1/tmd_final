import pandas as pd
import re, string, csv
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.manifold import TSNE

### ========== Step 1: Load Raw CSV ==========
INPUT_CSV = "bluesky_posts.csv"
OUTPUT_CSV = "bluesky_posts_with_topics.csv"

print(f"📥 Loading data from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

### ========== Step 2: Text Cleaning ==========

custom_stopwords = ENGLISH_STOP_WORDS.union({
    "new", "post", "hello", "just", "like", "people", "make", "good", "art", "day", "thing", "today", "blue", "sky"
})

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"http\\S+|@\\S+|#\\S+", " ", txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    return " ".join(w for w in txt.split() if w not in custom_stopwords and len(w) > 2)

print("🧹 Cleaning text...")
df["clean_text"] = df["text"].astype(str).apply(clean_text)

### ========== Step 3: Sentiment Analysis ==========

def safe_sentiment(text):
    if isinstance(text, str) and text.strip():
        return TextBlob(text).sentiment.polarity
    return 0.0

def sentiment_label(score):
    if score > 0.2:
        return "😊 Positive"
    elif score < -0.2:
        return "☹️ Negative"
    return "😐 Neutral"

print("🧠 Performing sentiment analysis...")
df["sentiment"] = df["text"].apply(safe_sentiment)
df["sentiment_label"] = df["sentiment"].apply(sentiment_label)

### ========== Step 4: Topic Modeling (LDA) ==========

print("📚 Running LDA topic modeling...")
vec = CountVectorizer(max_df=0.8, min_df=5)
X = vec.fit_transform(df["clean_text"])

k = 6
lda = LDA(n_components=k, random_state=0)
topic_dist = lda.fit_transform(X)
df["topic"] = topic_dist.argmax(axis=1)

def get_topic_labels(model, feature_names, n=5):
    return [" | ".join([feature_names[i] for i in topic.argsort()[:-n-1:-1]]) for topic in model.components_]

topic_labels = get_topic_labels(lda, vec.get_feature_names_out())
df["topic_label"] = df["topic"].apply(lambda x: topic_labels[x])

### ========== Step 5: Dimensionality Reduction (t-SNE) ==========

print("🔢 Performing t-SNE projection...")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=0)
tsne_results = tsne.fit_transform(topic_dist)
df["x"], df["y"] = tsne_results[:, 0], tsne_results[:, 1]

### ========== Step 6: Topic Sizes ==========
df["size"] = df["topic_label"].map(df["topic_label"].value_counts())

### ========== Step 7: Save ==========
columns_to_save = ["text", "topic_label", "sentiment", "sentiment_label", "x", "y", "size"]
df[columns_to_save].to_csv(OUTPUT_CSV, index=False)
print(f"✅ Data processed and saved to {OUTPUT_CSV}")

# # bluesky_lda_pipeline.py

# import requests
# from datetime import datetime, timedelta
# import json
# import csv
# import pandas as pd
# import re, string
# from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
# from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.manifold import TSNE
# from textblob import TextBlob

# # # Step 1: Crawl and save posts
# # url = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
# # hashtags = ["BlueskyMigration", "TwitterExodus", "Xodus",
# #             "FirstPost", 'NewToBluesky', 'ByeByeX', 'WhyBlueSky', 'BlueSkyTakeOver']

# # start_date = datetime(2023, 6, 1)
# # end_date = datetime(2025, 4, 30)

# # def format_bsky_datetime(dt):
# #     return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

# # all_posts, seen_uris = [], set()
# # current, previous_month = start_date, None

# # while current < end_date:
# #     next_week = current + timedelta(days=7)
# #     for tag in hashtags:
# #         params = {
# #             "q": tag, "sort": "latest", "limit": 100, "lang": "en",
# #             "since": format_bsky_datetime(current),
# #             "until": format_bsky_datetime(next_week)
# #         }
# #         try:
# #             response = requests.get(url, params=params)
# #             if response.status_code != 200:
# #                 print(f"Error {response.status_code}: {response.text}")
# #                 continue
# #             posts = response.json().get("posts", [])
# #             for post in posts:
# #                 uri = post.get("uri")
# #                 if uri and uri not in seen_uris:
# #                     seen_uris.add(uri)
# #                     all_posts.append(post)
# #         except Exception as e:
# #             print(f"Error for #{tag}: {e}")
# #     current = next_week

# # # Save to CSV
# # with open("bluesky_posts.csv", "w", newline='', encoding="utf-8") as f_csv:
# #     writer = csv.writer(f_csv)
# #     writer.writerow(["uri", "author", "createdAt", "indexedAt", "text", "replyCount", "repostCount", "likeCount", "quoteCount"])
# #     for post in all_posts:
# #         writer.writerow([
# #             post.get("uri", ""),
# #             post.get("author", {}).get("handle", ""),
# #             post.get("record", {}).get("createdAt", ""),
# #             post.get("indexedAt", ""),
# #             post.get("record", {}).get("text", "").replace("\n", " ").strip(),
# #             post.get("replyCount", 0), post.get("repostCount", 0),
# #             post.get("likeCount", 0), post.get("quoteCount", 0)
# #         ])

# # Step 2–5: Clean, analyze, model, embed, save

# # Read CSV
# df = pd.read_csv("bluesky_posts.csv")

# # Clean text
# custom_stopwords = ENGLISH_STOP_WORDS.union({
#     "new", "post", "hello", "just", "like", "people", "make", "good", "art", "day", "thing", "today", "blue", "sky"
# })

# def clean_text(txt):
#     txt = txt.lower()
#     txt = re.sub(r"http\S+|@\S+|#\S+", " ", txt)
#     txt = txt.translate(str.maketrans('', '', string.punctuation))
#     return " ".join(w for w in txt.split() if w not in custom_stopwords and len(w) > 2)

# df["clean_text"] = df["text"].astype(str).apply(clean_text)

# # Sentiment analysis
# def safe_sentiment(text):
#     if isinstance(text, str) and text.strip():
#         return TextBlob(text).sentiment.polarity
#     else:
#         return 0.0  # or float('nan')

# df["sentiment"] = df["text"].apply(safe_sentiment)

# # LDA topic modeling
# vec = CountVectorizer(max_df=0.8, min_df=5)
# X = vec.fit_transform(df["clean_text"])
# k = 6
# lda = LDA(n_components=k, random_state=0)
# topic_dist = lda.fit_transform(X)
# df["topic"] = topic_dist.argmax(axis=1)

# # Top topic labels
# def get_topic_labels(model, feature_names, n=5):
#     return [" | ".join([feature_names[i] for i in topic.argsort()[:-n-1:-1]]) for topic in model.components_]

# topic_labels = get_topic_labels(lda, vec.get_feature_names_out())
# df["topic_label"] = df["topic"].apply(lambda x: topic_labels[x])

# # t-SNE dimensionality reduction
# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=0)
# tsne_results = tsne.fit_transform(topic_dist)
# df["x"], df["y"] = tsne_results[:, 0], tsne_results[:, 1]

# # Topic size
# df["size"] = df["topic_label"].map(df["topic_label"].value_counts())

# # Save enriched CSV
# df[["text", "topic_label", "sentiment", "x", "y", "size"]].to_csv("bluesky_posts_with_topics.csv", index=False)
# print("✅ Data processed and saved to bluesky_posts_with_topics.csv")
