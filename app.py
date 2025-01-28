import streamlit as st
import PyPDF2
import speech_recognition as sr
import spacy
from transformers import pipeline
import google.generativeai as genai

# Set up the Gemini API key
genai.configure(api_key="your-api-key-here")

# Initialize spaCy model for NER
nlp = spacy.load("en_core_web_trf")

# Hugging Face Sentiment Analysis model
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# Functions for different tasks

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to transcribe audio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Audio could not be understood."
    except sr.RequestError as e:
        return f"Request failed: {e}"

# Function to extract entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function for Intent Detection using Google Gemini API
def detect_intent(text):
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
        f"Detect the intent of the following sentence: {text}"
    )
    return response.text.strip()

# Streamlit UI
st.title("Text, Speech, NER, Sentiment, and Intent Detection")

# Step 1: Choose Input Type
input_type = st.radio("Select Input Type", ["Text", "PDF", "Audio"])

if input_type == "Text":
    # Step 2: Text Input
    st.header("Enter Your Text")
    user_input_text = st.text_area("Enter text here")

    if user_input_text:
        # Sentiment Analysis
        sentiment = sentiment_analysis(user_input_text)
        st.write("Sentiment Analysis Result: ", sentiment)

        # Named Entity Recognition (NER)
        entities = extract_entities(user_input_text)
        st.write("Named Entities: ", entities)

        # Intent Detection
        intent = detect_intent(user_input_text)
        st.write(f"Detected Intent: {intent}")

elif input_type == "PDF":
    # Step 2: PDF Input
    st.header("Upload Your PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_pdf is not None:
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        st.write(pdf_text[:500])  # Display first 500 characters

        # NER on extracted PDF text
        entities = extract_entities(pdf_text)
        st.write("Named Entities: ", entities)

        # Sentiment Analysis on extracted PDF text
        sentiment = sentiment_analysis(pdf_text)
        st.write("Sentiment Analysis Result: ", sentiment)

        # Intent Detection on extracted PDF text
        intent = detect_intent(pdf_text)
        st.write(f"Detected Intent: {intent}")

elif input_type == "Audio":
    # Step 2: Audio Input
    st.header("Upload Your Audio File")
    uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_audio is not None:
        # Transcribe audio to text
        audio_text = transcribe_audio(uploaded_audio)
        st.write(audio_text[:500])  # Display first 500 characters of the transcribed text

        # NER on transcribed audio text
        entities = extract_entities(audio_text)
        st.write("Named Entities: ", entities)

        # Sentiment Analysis on transcribed audio text
        sentiment = sentiment_analysis(audio_text)
        st.write("Sentiment Analysis Result: ", sentiment)

        # Intent Detection on transcribed audio text
        intent = detect_intent(audio_text)
        st.write(f"Detected Intent: {intent}")
