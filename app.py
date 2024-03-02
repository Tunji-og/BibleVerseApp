import streamlit as st
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load the Bible data
data = pd.read_csv("./kjv.csv")

# Preprocess the Text data
stop_words = set(stopwords.words('english'))
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Text'].fillna(''))


# Function to get Bible verses based on user input
def get_verse_suggestions(user_input, num_suggestions=1):
    user_input_transformed = tfidf_vectorizer.transform([user_input])
    cosine_similarities = linear_kernel(user_input_transformed, tfidf_matrix).flatten()
    related_verse_indices = cosine_similarities.argsort()[:-num_suggestions-1:-1]
    suggested_verses = data.iloc[related_verse_indices]
    return suggested_verses[['Book Name', 'Chapter', 'Verse', 'Text']]

# Streamlit app
def main():
    st.title("Bible Verse Suggestion App")
    st.subheader("By Adetunji Ogunsusi")
    user_input = st.text_area("How are you feeling today?")
    
    if st.button("Get Bible Verse"):
        if user_input:
            st.subheader("Here is a verse for you:")
            suggestions = get_verse_suggestions(user_input)
            for _, verse in suggestions.iterrows():
                st.write(f"{verse['Book Name']} {verse['Chapter']}:{verse['Verse']} - {verse['Text']}")
        else:
            st.warning("Please enter your feelings to get a corresponding verse.")

if __name__ == "__main__":
    main()
