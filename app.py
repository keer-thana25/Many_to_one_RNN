import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="SMS Spam Detection", page_icon="📱")

st.title("📱 SMS Spam Detection")
st.write("This app uses a **Many-to-One RNN** to classify SMS messages as either Spam or Ham (legitimate).")

# Load model and tokenizer
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('model/spam_model.keras')
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_resources()
    # Get the sequence length the model expects
    maxlen = model.input_shape[1]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

user_input = st.text_area("Enter a message to check:", "Congratulations! You've won a $1000 gift card. Reply WIN to claim now.")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess the text
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=maxlen)
            
            # Predict
            pred = model.predict(padded)[0][0]
            
            if pred > 0.5:
                st.error(f"🚨 **SPAM DETECTED!**")
                st.write(f"Confidence: {pred:.1%}")
            else:
                st.success(f"✅ **Safe (Ham)**")
                st.write(f"Spam Probability: {pred:.1%}")