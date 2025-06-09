import streamlit as st
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the Bible data from the CSV file
data = pd.read_csv("./kjv.csv")

# Generate embeddings for all Bible verses once when the app starts
bible_verse_embeddings = model.encode(data['Text'].fillna('').tolist(), convert_to_tensor=True)

# --- New: Emotional Keyword Mapping (Optional but Recommended) ---
# This dictionary maps common user feelings to more focused biblical themes or keywords.
# You can expand this significantly.
# Expanded example of emotional_keyword_map
emotional_keyword_map = {
    "sad": ["comfort", "hope", "mourning", "sorrow", "peace", "grief", "tears"],
    "depressed": ["comfort", "hope", "strength", "light", "joy", "renewal", "peace"],
    "anxious": ["peace", "worry", "trust", "fear", "rest", "calm", "faith"],
    "stressed": ["peace", "rest", "burden", "strength", "calm", "release"],
    "happy": ["joy", "thanksgiving", "praise", "blessing", "rejoice", "gladness"],
    "lonely": ["presence", "companionship", "comfort", "never alone"],
    "lost": ["guidance", "path", "direction", "wisdom", "light", "truth"],
    "fearful": ["courage", "strength", "peace", "protection", "no fear", "boldness"],
    "grateful": ["thanksgiving", "praise", "blessing", "gratitude"],
    "angry": ["patience", "forgiveness", "peace", "calm", "self-control"],
    "confused": ["wisdom", "understanding", "guidance", "clarity"],
    "tired": ["rest", "strength", "renewal", "peace", "refreshment"],
    "overwhelmed": ["rest", "peace", "strength", "burden", "help", "calm"],
    "guilty": ["forgiveness", "redemption", "cleansing", "mercy", "grace"],
    "doubtful": ["faith", "trust", "guidance", "wisdom", "truth", "believe"],
    "frustrated": ["patience", "peace", "perseverance", "strength"],
    "joyful": ["joy", "praise", "thanksgiving", "rejoice", "gladness"],
    "weak": ["strength", "power", "help", "renewal"],
    "suffering": ["comfort", "endurance", "hope", "strength", "healing"],
    "insecure": ["confidence", "strength", "identity", "love", "acceptance"]
}

# Function to get Bible verses based on user input using semantic similarity
def get_verse_suggestions(user_input, num_suggestions=5): # Increased num_suggestions to give more options
    # --- Enhancement 1: Process User Input with Keyword Mapping ---
    # Convert user input to lowercase for consistent matching
    user_input_lower = user_input.lower()
    
    # Try to find a direct mapping to biblical keywords
    mapped_keywords = []
    for feeling, keywords in emotional_keyword_map.items():
        if feeling in user_input_lower:
            mapped_keywords.extend(keywords)

    # If specific keywords are found, use them to augment the search.
    # Otherwise, use the original user input.
    search_query = user_input
    if mapped_keywords:
        # Augment the user's input with relevant biblical keywords.
        # This helps steer the semantic search towards more relevant topics.
        search_query = user_input + " " + " ".join(mapped_keywords)
        st.info(f"Refining search with keywords: {', '.join(mapped_keywords)}") # Inform the user


    # Generate a semantic embedding for the refined search query
    user_input_embedding = model.encode([search_query], convert_to_tensor=True)

    # Calculate cosine similarities between the user's input embedding
    # and all Bible verse embeddings.
    cosine_similarities = cosine_similarity(user_input_embedding.cpu(), bible_verse_embeddings.cpu()).flatten()

    # Get the indices of the most similar verses (sorted in descending order of similarity)
    # Get top N suggestions initially
    top_n_indices = cosine_similarities.argsort()[-num_suggestions*3:][::-1] # Get more than needed for filtering

    # --- Enhancement 2: Simple Keyword Filter (Optional, but can improve relevance) ---
    # You might want to filter or re-rank these. For example, if the user mentioned "peace",
    # you might prefer verses that explicitly contain "peace" or related terms.
    
    filtered_indices = []
    
    # Define keywords for filtering (you can expand this logic)
    filter_keywords = mapped_keywords # Use the mapped keywords for filtering
    
    if filter_keywords:
        for idx in top_n_indices:
            verse_text = data.iloc[idx]['Text'].lower()
            # Check if any of the filter keywords are present in the verse text
            if any(keyword in verse_text for keyword in filter_keywords):
                filtered_indices.append(idx)
            if len(filtered_indices) >= num_suggestions: # Stop once we have enough filtered verses
                break
    
    # If filtering didn't yield enough results, or no keywords were mapped,
    # fall back to the purely semantic top N.
    if len(filtered_indices) < num_suggestions:
        # If we couldn't filter enough, just take the top semantic matches
        final_indices = cosine_similarities.argsort()[:-num_suggestions-1:-1]
    else:
        final_indices = filtered_indices[:num_suggestions]


    # Retrieve the suggested verses from the original DataFrame using the identified indices
    suggested_verses = data.iloc[final_indices]

    # Return only the relevant columns for display
    return suggested_verses[['Book Name', 'Chapter', 'Verse', 'Text']]

# Streamlit application main function
def main():
    st.title("Bible Verse Suggestion App")
    st.subheader("By Adetunji Ogunsusi")

    user_input = st.text_area("How are you feeling today?", placeholder="e.g., I'm feeling really sad and need some comfort.")

    if st.button("Get Bible Verse"):
        if user_input:
            st.subheader("Here are some verses for you:")
            suggestions = get_verse_suggestions(user_input, num_suggestions=3) # Suggest 3 verses
            if not suggestions.empty:
                for _, verse in suggestions.iterrows():
                    st.write(f"**{verse['Book Name']} {verse['Chapter']}:{verse['Verse']}** - {verse['Text']}")
                    st.write("---") # Add a separator for better readability
            else:
                st.info("No verses found for your feelings. Please try rephrasing.")
        else:
            st.warning("Please enter your feelings to get a corresponding verse.")

if __name__ == "__main__":
    main()