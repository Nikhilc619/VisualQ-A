import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Model Loading
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")

# Streamlit App Structure
st.title("Visual Question Answering ")

def get_image():
    img_url = st.text_input("Enter Image URL", value='https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg')
    if img_url:
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')  
        st.image(raw_image)
        return raw_image

def process_vqa(image, question):
    if image and question:
        inputs = processor(image, question, return_tensors="pt")
        output = model.generate(**inputs)
        answer = processor.decode(output[0], skip_special_tokens=True)
        st.write("Answer:", answer)

# User Input
image = get_image()
question = st.text_input("Ask your question about the image:")

# Process Question and Generate Answer
process_vqa(image, question) 
