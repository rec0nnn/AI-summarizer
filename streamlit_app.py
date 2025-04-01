import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import fitz  #PyMuPDF for PDF handling
import docx  #python-docx for DOCX handling

#Initialize the model and tokenizer
model_name = "t5-base" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

#Summarization function
def summarize_text(text, max_input_length=512, max_output_length=100):
    #Prepare the input for summarization
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)

    #Generate the summary with adjusted parameters
    summary_ids = model.generate(
        inputs,
        max_length=max_output_length,
        min_length=40,             #Minimum length for the summary to avoid very short results
        length_penalty=2.5,        #Higher length penalty encourages shorter summaries
        num_beams=4,               #Use fewer beams for faster but more varied results
        do_sample=True,            #Enables sampling for more diversity
        temperature=0.7,           #Controls the randomness of predictions
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


#Function to read different document types
def read_file(file):
    #TXT file
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    
    #PDF file
    elif file.type == "application/pdf":
        pdf_text = ""
        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            pdf_text += page.get_text("text")
        pdf_doc.close()
        return pdf_text
    
    #DOCX file
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        docx_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return docx_text

    else:
        st.error("Unsupported file type.")
        return None

#Summarize the document in chunks
def summarize_document(text):
    max_input_length = 512
    if len(text) > max_input_length:
        summaries = []
        for i in range(0, len(text), max_input_length):
            chunk = text[i:i+max_input_length]
            summary = summarize_text(chunk)
            summaries.append(summary)
        full_summary = " ".join(summaries)
    else:
        full_summary = summarize_text(text)
    return full_summary

#Streamlit UI
st.title("AI-Powered Document Summarizer")
st.write("Upload a text, PDF, or DOCX file, or paste text below to generate a summary.")

#Input method selection
input_method = st.radio("Choose input method:", ("Upload a file", "Paste text"))

text = ""
if input_method == "Upload a file":
    #File upload for txt, pdf, and docx
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        text = read_file(uploaded_file)
        if text:
            st.write("Original Document:")
            st.text_area(label="", value=text, height=200)
elif input_method == "Paste text":
    #Text area for pasting text
    text = st.text_area("Paste your text here:", height=200)

#Summarize the text when the button is clicked
if st.button("Summarize") and text:
    summary = summarize_document(text)
    st.write("Summary of the Document:")
    st.text_area(label="", value=summary, height=150)
