import re
from collections import Counter, defaultdict
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

def parse_chat_from_txt(filename, speaker_labels=None):
    if speaker_labels is None:
        speaker_labels = ['User', 'AI']
    pattern = re.compile(rf"^({'|'.join(speaker_labels)}):\s*(.*)$")
    messages = defaultdict(list)
    total_lines = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                speaker, text = m.group(1), m.group(2)
                messages[speaker].append(text)
                total_lines += 1
    return messages, total_lines

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

def extract_keywords(messages, n=None):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    all_text = " ".join(messages).lower()

    tokens = word_tokenize(all_text)
    filtered = [tok for tok in tokens if tok.isalpha() and tok not in stop_words]

    tagged = pos_tag(filtered)
    lemmas = [lemmatizer.lemmatize(tok, map_pos_to_wordnet(tag)) for tok, tag in tagged]

    freq = Counter(lemmas)
    return [word for word, _ in freq.most_common(n)]

def generate_summary(all_keywords):
    print(" Summary")
    print(f" - The conversation had {len(all_keywords)} exchanges")

    if len(all_keywords) >= 2:
        print(f" - The user asked mainly about {all_keywords[0]} and {all_keywords[1]}")
    elif len(all_keywords) == 1:
        print(f" - The user asked mainly about {all_keywords[0]}")
    else:
        print(" - Couldn't extract main topics from user.")

    print(f" - Most common keywords: {', '.join(all_keywords[:5]) if all_keywords else 'N/A'}")

if __name__ == "__main__":
    chat_file = "chat_log/chat.txt"  # Path to your chat file
    messages, total_exchanges = parse_chat_from_txt(chat_file)
    
    all_msgs = messages.get('User', []) + messages.get('AI', [])
    all_keywords = extract_keywords(all_msgs)
    print(f"\nTotal messages: {total_exchanges}")
    print(f"User messages: {len(messages.get('User', []))}")
    print(f"AI messages: {len(messages.get('AI', []))}\n")

    generate_summary(all_keywords)
