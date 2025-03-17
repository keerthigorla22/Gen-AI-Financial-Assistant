import pandas as pd
import re

# Define a simple stopwords list so we don’t depend on external downloads
manual_stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now'
])

def clean_text(text):
    """
    Lowercase the text, remove non-alphabetical characters,
    and remove stopwords.
    """
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in manual_stop_words)
    return text

# Load your dataset from the provided path
df = pd.read_csv("C:/Users/KEERTHI/Downloads/BankFAQs.csv")

# Remove any rows with missing values
df.dropna(inplace=True)

# Clean the 'Question' column
df['cleaned_text'] = df['Question'].apply(clean_text)

# Create a combined document that includes both question and answer
df['document_content'] = "Question: " + df['cleaned_text'] + "\nAnswer: " + df['Answer']

# Save the preprocessed data for reuse
df.to_csv("Preprocessed_BankFAQs.csv", index=False)
print("Preprocessing complete. Saved to Preprocessed_BankFAQs.csv")
