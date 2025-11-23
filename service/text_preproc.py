
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk
############# CLEANING DATA (UPDATED WITH NEGATION HANDLING) ##################

import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk

# Từ phủ định cần giữ lại
NEGATION_WORDS = {
    "not","no","nor","never","without",
    "n't", "dont","don't","cant","can't","cannot","wont","won't",
    "isnt","isn't","arent","aren't","wasnt","wasn't",
    "werent","weren't","shouldnt","shouldn't","wouldnt","wouldn't",
    "couldnt","couldn't","mustnt","mustn't"
}

def handle_negation(tokens):
    new_tokens = []
    neg_left = 0

    for tok in tokens:
        # Nếu token đã NEG_ rồi thì giữ nguyên
        if tok.startswith("NEG_"):
            new_tokens.append(tok)
            continue

        if tok in NEGATION_WORDS:
            neg_left = 2
            new_tokens.append(tok)
        elif neg_left > 0:
            new_tokens.append("NEG_" + tok)
            neg_left -= 1
        else:
            new_tokens.append(tok)

    return new_tokens



def clean_text(text):
    # 1) Xóa HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'<.*?>', '', text)

    # 2) Lowercase
    text = text.lower()

    # 3) Giữ lại chữ cái (có hỗ trợ tiếng Việt)
    text = re.sub(r"[^a-zA-Zà-ỹÀ-Ỹ\s]", " ", text)

    # 4) Xóa khoảng trắng dư
    text = re.sub(r"\s+", " ", text).strip()

    # 5) Tokenize
    tokens = text.split()

    # 6) Loại stopwords TRỪ TỪ PHỦ ĐỊNH
    stop_words = set(ENGLISH_STOP_WORDS)
    stop_words = stop_words.difference(NEGATION_WORDS)
    tokens = [t for t in tokens if t not in stop_words]

    # 7) ÁP DỤNG NEGATION HANDLING  🔥
    tokens = handle_negation(tokens)

    # 8) Stemming
    stemmer = nltk.stem.SnowballStemmer("english")
    tokens = [stemmer.stem(t) for t in tokens]

    # 9) Ghép lại
    return " ".join(tokens)

def _clean_batch(X):
    return np.array([clean_text(t) for t in X], dtype=object)
