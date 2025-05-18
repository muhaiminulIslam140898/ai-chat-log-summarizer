import re
import nltk
from collections import Counter
from nltk.corpus import stopwords

nltk.download('stopwords')

def read_chat_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def parse_chat(lines):
    user_msgs = []
    ai_msgs = []
    for line in lines:
        if line.startswith("User:"):
            user_msgs.append(line[len("User:"):].strip())
        elif line.startswith("AI:"):
            ai_msgs.append(line[len("AI:"):].strip())
    return user_msgs, ai_msgs

def get_message_stats(user_msgs, ai_msgs):
    total = len(user_msgs) + len(ai_msgs)
    return total, len(user_msgs), len(ai_msgs)

def preprocess_and_tokenize(messages):
    stops = set(stopwords.words('english'))
    words = []
    for msg in messages:
        tokens = re.findall(r'\b\w+\b', msg.lower())
        filtered = [w for w in tokens if w not in stops]
        words.extend(filtered)
    return words

def get_top_keywords(user_msgs, ai_msgs, top_n=5):
    words = preprocess_and_tokenize(user_msgs + ai_msgs)
    counts = Counter(words)
    return counts.most_common(top_n)

def generate_summary(total, user_count, ai_count, keywords):
    print("----- Chat Summary -----")
    print(f"Total exchanges : {total}")
    print(f"User messages   : {user_count}")
    print(f"AI messages     : {ai_count}")
    if keywords:
        print("Top keywords    :", ", ".join([kw for kw, _ in keywords]))
    else:
        print("No keywords found.")

def summarize_chat(file_path):
    lines = read_chat_file(file_path)
    user_msgs, ai_msgs = parse_chat(lines)
    total, user_count, ai_count = get_message_stats(user_msgs, ai_msgs)
    keywords = get_top_keywords(user_msgs, ai_msgs)
    generate_summary(total, user_count, ai_count, keywords)

if __name__ == "__main__":
    summarize_chat("chat.txt")
