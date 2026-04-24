import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Drug Review Sentiment Analyser",
    page_icon="💊",
    layout="wide"
)

# ── Load data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/df_app.csv.gz', compression='gzip')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.error("Data failed to load. Check the data file exists.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💊 About This App")
    st.markdown("""
    This app analyses patient drug reviews using:
    - **NLP sentiment analysis** — DistilBERT transformer 
      fine-tuned on 4,000 drug reviews
    - **TF-IDF theme extraction** — identifies distinctive 
      topics in positive and negative reviews
    - **215,000+ reviews** from Drugs.com (2008-2017)
    """)
    st.divider()
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Reviews", f"{len(df):,}")
    st.metric("Unique Drugs", f"{df['drugName'].nunique():,}")
    st.metric("Unique Conditions", f"{df['condition'].nunique():,}")
    st.divider()
    st.markdown("""
    *Data source: Drugs.com via UCI ML Repository*
    
    ⚠️ *For analytical purposes only. Does not constitute 
    medical advice.*
    """)


# ── Header ───────────────────────────────────────────────────────
st.title("💊 Drug Review Sentiment Analyser")
st.markdown("""
Patient-reported experiences are one of the most underutilised 
sources of insight in drug safety and pharmacovigilance. This app 
analyses **215,000+ patient reviews** to surface sentiment patterns 
and key themes for any drug or condition.
""")
st.divider()

# ── Search ───────────────────────────────────────────────────────
st.subheader("🔍 Search by Drug or Condition")

col1, col2 = st.columns([3, 1])

with col1:
    search_term = st.text_input(
        label="Enter a drug name or condition",
        placeholder="e.g. metformin, depression, birth control..."
    )

with col2:
    search_type = st.radio(
        "Search by:",
        options=["Drug", "Condition"]
    )

# ── Filter data based on search ──────────────────────────────────
if search_term:
    if search_type == "Drug":
        results = df[df['drugName'].str.contains(
            search_term, case=False, na=False)]
    else:
        results = df[df['condition'].str.contains(
            search_term, case=False, na=False)]

    # Show result count
    if len(results) == 0:
        st.warning(f"No reviews found for '{search_term}'. "
                   f"Please try a different search term.")
        st.stop()
    else:
        st.success(f"Found **{len(results):,} reviews** "
                   f"for '{search_term}'")
else:
    st.info("👆 Enter a drug name or condition above to begin")
    st.stop()

# ── Sentiment Distribution ───────────────────────────────────────
st.divider()
st.subheader("📊 Sentiment Distribution")

# Calculate sentiment counts and percentages
sentiment_counts = results['sentiment'].value_counts()
sentiment_pct = (results['sentiment'].value_counts(normalize=True) * 100).round(1)

# Display metrics in columns
col1, col2, col3 = st.columns(3)

with col1:
    pos_count = sentiment_counts.get('positive', 0)
    pos_pct = sentiment_pct.get('positive', 0)
    st.metric(
        label="✅ Positive",
        value=f"{pos_count:,}",
        delta=f"{pos_pct}% of reviews"
    )

with col2:
    neg_count = sentiment_counts.get('negative', 0)
    neg_pct = sentiment_pct.get('negative', 0)
    st.metric(
        label="❌ Negative",
        value=f"{neg_count:,}",
        delta=f"{neg_pct}% of reviews",
        delta_color="inverse"
    )

with col3:
    neu_count = sentiment_counts.get('neutral', 0)
    neu_pct = sentiment_pct.get('neutral', 0)
    st.metric(
        label="➖ Neutral",
        value=f"{neu_count:,}",
        delta=f"{neu_pct}% of reviews",
        delta_color="off"
    )

# Sentiment bar chart
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#2ecc71', '#e74c3c', '#95a5a6']
bars = ax.bar(
    ['Positive', 'Negative', 'Neutral'],
    [pos_count, neg_count, neu_count],
    color=colors,
    edgecolor='black',
    linewidth=0.5
)

# Add percentage labels on bars
for bar, pct in zip(bars, [pos_pct, neg_pct, neu_pct]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{pct}%',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

ax.set_title(f'Sentiment Distribution for "{search_term}"',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Reviews')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()


# ── Theme Extraction ─────────────────────────────────────────────
st.divider()
st.subheader("🏷️ Key Themes")

def extract_themes(texts, n_terms=10):
    if len(texts) < 10:
        return pd.DataFrame(columns=['term', 'score'])
    
    custom_stop_words = [
        'drug', 'medication', 'medicine', 'doctor', 'prescribed',
        'taking', 'taken', 'take', 'took', 'tablet', 'pill', 'pills',
        'day', 'days', 'week', 'weeks', 'month', 'months', 'year',
        'years', 'time', 'just', 'really', 'got', 'get', 'like',
        'know', 'feel', 'felt', 'think', 'tried', 'try', 'started',
        'start', 'use', 'used', 'using', 'told', 'said', 'went',
        'going', 'came', 'come', 'good', 'great', 'bad', 'better',
        'worse', 'best', 'lot', 'little', 'bit', 'way', 'back',
        'well', 'work', 'works', 'worked', 'help', 'helps', 'helped', 
        've', 'did', 'don', 'far', 'didn', 'having', 'night', 'ago', 
        'dose', 'hours', 'symptoms', 'does'
    ]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        terms = vectorizer.get_feature_names_out()
        term_scores = pd.DataFrame({'term': terms, 'score': mean_scores})
        term_scores = term_scores[
            ~term_scores['term'].str.contains(
                '|'.join(custom_stop_words), case=False)]
        return term_scores.nlargest(n_terms, 'score')
    except:
        return pd.DataFrame(columns=['term', 'score'])

# Extract themes for positive and negative
pos_reviews = results[results['sentiment'] == 'positive']['review_clean']
neg_reviews = results[results['sentiment'] == 'negative']['review_clean']

pos_themes = extract_themes(pos_reviews)
neg_themes = extract_themes(neg_reviews)

# Display themes side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ✅ What patients praise")
    if not pos_themes.empty:
        for _, row in pos_themes.iterrows():
            st.markdown(f"- {row['term']}")
    else:
        st.info("Not enough positive reviews for theme extraction")

with col2:
    st.markdown("#### ❌ What patients complain about")
    if not neg_themes.empty:
        for _, row in neg_themes.iterrows():
            st.markdown(f"- {row['term']}")
    else:
        st.info("Not enough negative reviews for theme extraction")

# ── Most Useful Reviews ──────────────────────────────────────────
st.divider()
st.subheader("💬 Most Helpful Patient Reviews")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ✅ Top Positive Reviews")
    top_positive = (results[results['sentiment'] == 'positive']
                   .nlargest(3, 'usefulCount')[
                       ['review', 'usefulCount', 'drugName', 'condition']])
    
    if len(top_positive) > 0:
        for _, row in top_positive.iterrows():
            with st.expander(
                f"👍 {row['drugName']} — {row['condition']} "
                f"({row['usefulCount']} found helpful)"):
                st.write(row['review'])
    else:
        st.info("No positive reviews found")

with col2:
    st.markdown("#### ❌ Top Negative Reviews")
    top_negative = (results[results['sentiment'] == 'negative']
                   .nlargest(3, 'usefulCount')[
                       ['review', 'usefulCount', 'drugName', 'condition']])
    
    if len(top_negative) > 0:
        for _, row in top_negative.iterrows():
            with st.expander(
                f"👎 {row['drugName']} — {row['condition']} "
                f"({row['usefulCount']} found helpful)"):
                st.write(row['review'])
    else:
        st.info("No negative reviews found")

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: grey; font-size: 0.85em;'>
    <p>Built by Amrita Shah | Drug Review Sentiment Analyser</p>
    <p>Data source: Drugs.com Patient Reviews via UCI ML Repository</p>
    <p>⚠️ This tool is for analytical purposes only and does not 
    constitute medical advice. Patient data is presented with 
    appropriate clinical sensitivity.</p>
</div>
""", unsafe_allow_html=True)

