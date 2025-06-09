import pandas as pd
import streamlit as st

# Load the Bible verses CSV file
data = pd.read_csv("./kjv.csv")  # Replace with the actual file path

# Streamlit App
st.title("Bible Verse App")

# User Input
feeling = st.text_input("How are you feeling today?")

# Search for a corresponding Bible verse
if feeling:
    matching_verse = data[data['Text'].str.contains(feeling, case=False)].sample(1)
    if not matching_verse.empty:
        st.subheader("Corresponding Bible Verse:")
        st.write(matching_verse[['Book Name', 'Chapter', 'Verse', 'Text']].values[0])
    else:
        st.info("No matching Bible verse found for the provided feeling.")

# Display the entire DataFrame (optional)
if st.checkbox("Show Entire Bible Data"):
    st.write(data)
