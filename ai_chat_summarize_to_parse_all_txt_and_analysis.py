import os
import re
import string
from collections import Counter

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Config ─────────────────────────────────────────────────────────────────────
CHAT_FOLDER  = 'chat_log'      # folder containing .txt chat files
PATTERN      = r'(User|AI):\s*(.*?)(?=\n*User:|\n*AI:|$)'
STOPWORDS    = set(stopwords.words('english'))
LEMMATIZER   = WordNetLemmatizer()
PUNCT_TRANS  = str.maketrans('', '', string.punctuation)

# ─── POS Mapping Helper ─────────────────────────────────────────────────────────
def map_pos_to_wordnet(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

# ─── Functions ────────────────────────────────────────────────────────────────────
def extract_messages(text):
    return re.findall(PATTERN, text, re.DOTALL)

def preprocess(text):
    text = text.translate(PUNCT_TRANS).lower() # Remove punctuation and lowercase
    tokens = word_tokenize(text) # Tokenize using NLTK word Tokenizer
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS] # Remove stopwords
    tagged = pos_tag(tokens) # POS tagging
    lemmas = [
        LEMMATIZER.lemmatize(tok, map_pos_to_wordnet(pos_tag))
        for tok, pos_tag in tagged
    ]
    return lemmas

def top_tf_idf(docs, top_n=None):
    vec = TfidfVectorizer()
    tfidf_matrix = vec.fit_transform([" ".join(d) for d in docs])
    features = vec.get_feature_names_out()
    results = []
    for row in tfidf_matrix.toarray():
        scored = sorted(zip(features, row), key=lambda x: -x[1])
        results.append([w for w, _ in scored[:top_n]])
    return results

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Read & extract all messages
    msgs = []
    for fname in os.listdir(CHAT_FOLDER):
        if not fname.endswith('.txt'):
            continue
        text = open(os.path.join(CHAT_FOLDER, fname), encoding='utf-8').read()
        for speaker, msg in extract_messages(text):
            msgs.append((speaker, msg.strip()))

    user_msgs = [msg for speaker, msg in msgs if speaker == 'User']
    ai_msgs = [msg for speaker, msg in msgs if speaker == 'AI']
    combined_text = " ".join(user_msgs + ai_msgs)

    print(f'Total messages: {len(user_msgs) + len(ai_msgs)}')
    print(f'User messages: {len(user_msgs)}')
    print(f'AI messages: {len(ai_msgs)}\n')

    combined_tokens = preprocess(combined_text)
    combined_top = top_tf_idf([combined_tokens])[0]

    # Final Summary
    print("Summary")
    print(f"The conversation had {len(combined_top)} exchanges")
    if len(combined_top) >= 2:
        print(f"The user asked mainly about {combined_top[0]} and {combined_top[1]}")
    elif combined_top:
        print(f"The user asked mainly about {combined_top[0]}")
    print(f"Most common keywords: {', '.join(combined_top[:5])}\n")

if __name__ == '__main__':
    main()
