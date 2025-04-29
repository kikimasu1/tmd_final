"""Bluesky LDA pipeline – now returns the **full original fields PLUS analytics**
=======================================================================
The final CSV is essentially your raw `bluesky_posts.csv` augmented with:
• `clean_text` (hidden from output)
• `sentiment`, `sentiment_label`
• `topic_id`, `topic_label`, 2‑D embedding `x`,`y`, bubble `size`
• `date` (YYYY‑MM‑DD) parsed from `createdAt`

So downstream dashboards can access every original column (author, counts, etc.) **and** all the derived fields in one place.
"""

import pandas as pd, re, string, emoji, nltk, math
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.manifold import TSNE

# ------------------------------------------------------------------
# Setup & IO paths
# ------------------------------------------------------------------
INPUT_CSV  = "bluesky_posts.csv"
OUTPUT_CSV = "bluesky_posts_with_topics.csv"
print(f"📥 Loading {INPUT_CSV}")
raw_df = pd.read_csv(INPUT_CSV)             # keep a pristine copy for final merge
proc_df = raw_df.copy()                     # working copy we mutate

# ------------------------------------------------------------------
# Quick NLTK resource check
# ------------------------------------------------------------------
for pkg in ["wordnet", "punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# ------------------------------------------------------------------
# 1️⃣ Corpus‑driven stop‑words + cleaning helpers
# ------------------------------------------------------------------
RAW_STOP = set(ENGLISH_STOP_WORDS)
EXTRA_NOISE = {"www", "http", "https", "amp", "rt"}
TOKEN_RE = re.compile(r"[a-z]{3,15}")

all_tokens = set()
for txt in proc_df["text"].dropna():
    all_tokens.update(TOKEN_RE.findall(str(txt).lower()))
DOC_FREQ = pd.Series(list(all_tokens)).value_counts(normalize=True)
CORPUS_STOP = set(DOC_FREQ[DOC_FREQ > 0.5].index)
STOPWORDS = RAW_STOP | EXTRA_NOISE | CORPUS_STOP

NOISE_RE = [re.compile(r"^(.)\1{2,}$"), re.compile(r"\d"), re.compile(r".{20,}")]
lemmatizer = WordNetLemmatizer()
POS_MAP = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}

def wn_pos(w):
    tag = nltk.pos_tag([w])[0][1][0].upper()
    return POS_MAP.get(tag, wordnet.NOUN)

def noisy(tok:str)->bool:
    return any(p.search(tok) for p in NOISE_RE)

def clean_text(txt:str)->str:
    if not isinstance(txt,str):
        return ""
    txt = re.sub(r"https?://\S+","",txt.lower())
    txt = re.sub(r"@\S+","",txt)
    txt = emoji.replace_emoji(txt,"")
    words = nltk.word_tokenize(txt.translate(str.maketrans('', '', string.punctuation)))
    lemmas=[lemmatizer.lemmatize(w,wn_pos(w)) for w in words if w not in STOPWORDS and not noisy(w)]
    hashtags=re.findall(r"#(\w+)",txt)
    return " ".join(lemmas+hashtags+hashtags)  # duplicate hashtags for weight

print("🧹 Cleaning text …")
proc_df["clean_text"] = proc_df["text"].apply(clean_text)

# ------------------------------------------------------------------
# 2️⃣ Emoji‑aware sentiment
# ------------------------------------------------------------------
POS_EMJ="😊😁😀😄😃👍❤️💕🎉👏✨🥰"; NEG_EMJ="😔😕😢😭😠😡👎💔😞😟😩😤"

def sentiment(txt):
    """Robust polarity: works even if `txt` is NaN/float/etc. and never throws."""
    if txt is None or (isinstance(txt, float) and math.isnan(txt)):
        txt = ""
    txt = str(txt)  # guarantee string
    if not txt.strip():
        return 0.0
    try:
        base = TextBlob(txt).sentiment.polarity
    except Exception:
        base = 0.0
    adj  = sum(0.1 for e in POS_EMJ if e in txt) - sum(0.1 for e in NEG_EMJ if e in txt)
    return max(min(base + adj, 1.0), -1.0)
LABELS=[(-1.01,-0.3,"😠 Very Negative"),(-0.3,-0.1,"🙁 Negative"),(-0.1,0.1,"😐 Neutral"),(0.1,0.3,"🙂 Positive"),(0.3,1.01,"😊 Very Positive")]

def label(s):
    for lo,hi,lbl in LABELS:
        if lo<s<=hi:
            return lbl

print("🧠 Sentiment …")
proc_df["sentiment"]       = proc_df["text"].apply(sentiment)
proc_df["sentiment_label"] = proc_df["sentiment"].apply(label)

# ------------------------------------------------------------------
# 3️⃣ LDA topics (10)
# ------------------------------------------------------------------
vec = CountVectorizer(max_df=0.85,min_df=3)
X   = vec.fit_transform(proc_df["clean_text"])
VOC = vec.get_feature_names_out()
lda = LDA(n_components=10,learning_method='online',random_state=42)
T   = lda.fit_transform(X)
proc_df["topic_id"] = T.argmax(1)

def top_terms(idx,n=6):
    comp=lda.components_[idx]
    return [VOC[i] for i in comp.argsort()[-15:][::-1] if not noisy(VOC[i])][:n]

def human(terms):
    kw=[t.lower() for t in terms]
    if any(k in kw for k in ["firstpost","first"]):
        return "First‑time Posts"
    if any(k in kw for k in ["twitter","twitterexodus","xodus","migration"]):
        return "Leaving Twitter"
    if any(k in kw for k in ["invite","code"]):
        return "Requesting Invite Codes"
    return " / ".join(kw[:3])
ID2LAB={i:human(top_terms(i)) for i in range(10)}
proc_df["topic_label"] = proc_df["topic_id"].map(ID2LAB)

# ------------------------------------------------------------------
# 4️⃣ 2‑D embedding + bubble size
# ------------------------------------------------------------------
print("🔢 t‑SNE …")
Y=TSNE(n_components=2,perplexity=40,learning_rate=200,n_iter=2000,random_state=42).fit_transform(T)
proc_df["x"],proc_df["y"] = Y[:,0],Y[:,1]
proc_df["topic_count"] = proc_df["topic_label"].map(proc_df["topic_label"].value_counts())
eng = raw_df[["replyCount","repostCount","likeCount"]].fillna(0).astype(int).sum(1)
scale = 1+49*eng.div(max(eng) if eng.max()>0 else 1)
proc_df["size"] = proc_df["topic_count"]*scale.div(50).clip(lower=1)

# ------------------------------------------------------------------
# 5️⃣ Date parsing & hashtags
# ------------------------------------------------------------------
print("📅 Dates …")
proc_df["parsed_date"] = pd.to_datetime(raw_df["createdAt"],errors='coerce')
proc_df["date"]        = proc_df["parsed_date"].dt.date
proc_df["hashtags"]    = raw_df["text"].str.lower().str.findall(r"#\w+")

# ------------------------------------------------------------------
# 6️⃣ Merge → full enriched frame
# ------------------------------------------------------------------
print("🔗 Merging analytics back onto raw columns …")
KEEP_ANALYTICS = ["sentiment","sentiment_label","topic_id","topic_label","x","y","size","date"]
full_df = raw_df.join(proc_df[KEEP_ANALYTICS])

print(f"💾 Saving → {OUTPUT_CSV}")
full_df.to_csv(OUTPUT_CSV,index=False)
print("✅ Done – enriched CSV includes ALL original fields + analytics!")


# import pandas as pd
# import re, string, csv
# from textblob import TextBlob
# from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
# from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.manifold import TSNE
# import emoji
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet

# # Download necessary NLTK data
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('taggers/averaged_perceptron_tagger')
# except LookupError:
#     nltk.download('averaged_perceptron_tagger')

# ### ========== Step 1: Load Raw CSV ==========
# INPUT_CSV = "bluesky_posts.csv"
# OUTPUT_CSV = "bluesky_posts_with_topics.csv"

# print(f"📥 Loading data from {INPUT_CSV}")
# df = pd.read_csv(INPUT_CSV)

# ### ========== Step 2: Enhanced Text Cleaning ==========

# # Improved stopwords - remove terms that are actually meaningful in this context
# # and add platform-specific terms that don't add meaning
# bluesky_specific_stopwords = {
#     "bluesky", "bsky", "post", "posts", "just", "like", "get", "one", "going", 
#     "know", "time", "got", "see", "today", "day", "now", "really", "make", 
#     "made", "making", "find", "finding", "need", "new", "good", "still", 
#     "much", "way", "go", "going", "back", "think", "getting", "come", "coming",
#     "got", "getting", "going"
# }

# # Remove some words from ENGLISH_STOP_WORDS that might be meaningful in our context
# meaningful_words = {
#     "twitter", "app", "social", "media", "platform", "feed", "user", "users",
#     "community", "communities", "exodus", "migration"
# }

# custom_stopwords = ENGLISH_STOP_WORDS.union(bluesky_specific_stopwords) - meaningful_words

# # Get a lemmatizer to reduce words to their base form
# lemmatizer = WordNetLemmatizer()

# def get_wordnet_pos(word):
#     """Map POS tag to WordNet POS tag format"""
#     tag = nltk.pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ,
#                 "N": wordnet.NOUN,
#                 "V": wordnet.VERB,
#                 "R": wordnet.ADV}
#     return tag_dict.get(tag, wordnet.NOUN)

# def extract_hashtags(text):
#     """Extract hashtags from text to preserve them as features"""
#     if not isinstance(text, str):
#         return []
#     hashtag_pattern = r'#(\w+)'
#     return re.findall(hashtag_pattern, text.lower())

# def clean_text(txt):
#     """Enhanced text cleaning with hashtag preservation and lemmatization"""
#     if not isinstance(txt, str):
#         return ""
    
#     # Extract hashtags before cleaning
#     hashtags = extract_hashtags(txt)
#     hashtag_text = " ".join(hashtags)
    
#     # Convert to lowercase
#     txt = txt.lower()
    
#     # Remove URLs
#     txt = re.sub(r'https?://\S+', '', txt)
    
#     # Remove mentions
#     txt = re.sub(r'@\S+', '', txt)
    
#     # Remove punctuation
#     txt = txt.translate(str.maketrans('', '', string.punctuation))
    
#     # Remove emojis
#     txt = emoji.replace_emoji(txt, replace='')
    
#     # Tokenize
#     words = nltk.word_tokenize(txt)
    
#     # Lemmatize words based on their POS
#     lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) 
#                         for word in words 
#                         if word not in custom_stopwords and len(word) > 2]
    
#     # Combine cleaned text with preserved hashtags (giving hashtags more weight)
#     return " ".join(lemmatized_words) + " " + hashtag_text + " " + hashtag_text

# print("🧹 Cleaning text with enhanced methods...")
# df["clean_text"] = df["text"].apply(clean_text)

# ### ========== Step 3: Improved Sentiment Analysis ==========

# def analyze_sentiment(text):
#     """Improved sentiment analysis with emoji consideration"""
#     if not isinstance(text, str) or not text.strip():
#         return 0.0
    
#     # Count positive and negative emojis
#     positive_emojis = ['😊', '😁', '😀', '😄', '😃', '👍', '❤️', '💕', '🎉', '👏', '✨', '🥰']
#     negative_emojis = ['😔', '😕', '😢', '😭', '😠', '😡', '👎', '💔', '😞', '😟', '😩', '😤']
    
#     emoji_sentiment = 0
#     for emoji_char in positive_emojis:
#         emoji_sentiment += 0.1 * text.count(emoji_char)
#     for emoji_char in negative_emojis:
#         emoji_sentiment -= 0.1 * text.count(emoji_char)
    
#     # Get TextBlob sentiment
#     textblob_sentiment = TextBlob(text).sentiment.polarity
    
#     # Combine both sentiments
#     combined_sentiment = textblob_sentiment + emoji_sentiment
    
#     # Ensure it stays within -1 to 1 range
#     return max(min(combined_sentiment, 1.0), -1.0)

# def sentiment_label(score):
#     """More nuanced sentiment labeling"""
#     if score > 0.3:
#         return "😊 Very Positive"
#     elif score > 0.1:
#         return "🙂 Positive"
#     elif score < -0.3:
#         return "😠 Very Negative"
#     elif score < -0.1:
#         return "🙁 Negative"
#     return "😐 Neutral"

# print("🧠 Performing improved sentiment analysis...")
# df["sentiment"] = df["text"].apply(analyze_sentiment)
# df["sentiment_label"] = df["sentiment"].apply(sentiment_label)

# ### ========== Step 4: Topic Modeling Enhancement ==========

# print("📚 Running improved LDA topic modeling...")
# # Increase min_df to filter out very rare terms and decrease max_df to filter out very common terms
# vec = CountVectorizer(max_df=0.7, min_df=3, max_features=5000)
# doc_term_matrix = vec.fit_transform(df["clean_text"])
# feature_names = vec.get_feature_names_out()

# # Increase number of topics for more granularity
# num_topics = 10
# lda = LDA(
#     n_components=num_topics,
#     max_iter=20,       # More iterations for better convergence
#     learning_method='online',
#     random_state=42,
#     n_jobs=-1          # Use all available cores
# )
# topic_dist = lda.fit_transform(doc_term_matrix)
# df["topic"] = topic_dist.argmax(axis=1)

# # Extract more descriptive topic labels using top terms and relevance metric
# def get_improved_topic_labels(model, feature_names, n=6):
#     """Get more descriptive topic labels using word relevance metrics"""
#     # Get top N terms for each topic sorted by importance
#     topic_labels = []
#     for topic_idx, topic in enumerate(model.components_):
#         # Sort terms by relevance
#         topic_terms = [(feature_names[i], topic[i]) for i in range(len(feature_names))]
#         topic_terms.sort(key=lambda x: x[1], reverse=True)
        
#         # Filter out very generic terms
#         terms = [term for term, _ in topic_terms[:15] if term not in bluesky_specific_stopwords][:n]
        
#         if len(terms) < 2:  # If we don't have enough meaningful terms
#             terms = [feature_names[i] for i in topic.argsort()[:-n-1:-1]]
        
#         topic_label = " | ".join(terms)
#         topic_labels.append(topic_label)
    
#     return topic_labels

# topic_labels = get_improved_topic_labels(lda, feature_names)
# print("Generated topic labels:")
# for i, label in enumerate(topic_labels):
#     print(f"Topic {i}: {label}")

# df["topic_label"] = df["topic"].apply(lambda x: topic_labels[x])

# # Calculate topic coherence scores to assess quality
# topic_words = {}
# for topic, comp in enumerate(lda.components_):
#     word_idx = comp.argsort()[:-10-1:-1]
#     topic_words[topic] = [feature_names[i] for i in word_idx]

# print("\nTopic words:")
# for topic, words in topic_words.items():
#     print(f"Topic {topic}: {', '.join(words[:10])}")

# ### ========== Step 5: Dimensionality Reduction (t-SNE) ==========

# print("🔢 Performing t-SNE projection with improved parameters...")
# tsne = TSNE(
#     n_components=2,
#     perplexity=40,      # Increased perplexity for better global structure
#     learning_rate=200,
#     n_iter=2000,        # More iterations for better convergence
#     random_state=42
# )
# tsne_results = tsne.fit_transform(topic_dist)
# df["x"], df["y"] = tsne_results[:, 0], tsne_results[:, 1]

# ### ========== Step 6: Enhanced Topic Sizes ==========

# # Calculate topic size based on number of posts and engagement
# df["topic_count"] = df["topic_label"].map(df["topic_label"].value_counts())
# df["engagement"] = df["replyCount"].astype(int) + df["repostCount"].astype(int) + df["likeCount"].astype(int)

# # Normalize engagement to be between 1 and 50 for better visualization
# if df["engagement"].max() > 0:
#     df["norm_engagement"] = 1 + 49 * (df["engagement"] / df["engagement"].max())
# else:
#     df["norm_engagement"] = 1

# # Create a size metric that combines post count and engagement
# df["size"] = df["topic_count"] * df["norm_engagement"].apply(lambda x: max(x/50, 1))

# ### ========== Step 7: Add Metadata ==========

# # Replace the date parsing section in bluesky_lda_pipeline.py with this code:

# ### ========== Step 7: Add Metadata ==========

# # Extract creation date for time analysis
# print("📅 Adding date metadata...")

# # Use a more flexible datetime parser that handles various formats
# def parse_date_safely(date_string):
#     if not isinstance(date_string, str) or not date_string:
#         return None
    
#     try:
#         # Try parsing ISO 8601 format with various microsecond and timezone formats
#         return pd.to_datetime(date_string, errors='coerce')
#     except:
#         return None

# # Parse dates with flexible format handling
# df['parsed_date'] = df['createdAt'].apply(parse_date_safely)

# # Extract date components if parsing succeeded
# if df['parsed_date'].notna().any():
#     df['date'] = df['parsed_date'].dt.date
#     df['month'] = df['parsed_date'].dt.month_name()
#     print(f"✅ Successfully parsed dates for {df['parsed_date'].notna().sum()} of {len(df)} posts")
# else:
#     print("⚠️ Warning: Could not parse any dates from the createdAt column")
#     # Create dummy date columns to prevent errors
#     df['date'] = pd.to_datetime("2023-07-01").date()
#     df['month'] = "July"

# # Extract original hashtags for filtering
# def extract_all_hashtags(text):
#     if not isinstance(text, str):
#         return []
#     hashtag_pattern = r'#(\w+)'
#     return [f"#{tag.lower()}" for tag in re.findall(hashtag_pattern, text.lower())]

# df["hashtags"] = df["text"].apply(extract_all_hashtags)

# print(f"✅ Data processed and saved to {OUTPUT_CSV}")

# print("\nCongratulations! Your Bluesky analysis pipeline has been significantly improved.")
# print("The enhanced pipeline now provides:")
# print("1. Better text cleaning with lemmatization and hashtag preservation")
# print("2. More accurate sentiment analysis that considers emojis")
# print("3. More meaningful topic modeling with {} topics".format(num_topics))
# print("4. Improved visualization data with engagement metrics")
# print("5. Additional metadata for filtering and analysis")

