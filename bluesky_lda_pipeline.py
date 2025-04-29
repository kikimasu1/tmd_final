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
