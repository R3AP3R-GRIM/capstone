import easyocr
import os
import spacy
import spacy.cli
spacy.cli.download('en_core_web_lg')
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from transformers import LongformerTokenizer, LongformerForQuestionAnswering
from transformers import LongformerTokenizerFast
import torch
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

with st.sidebar:
    selected=option_menu('Capstone Project- Exam Helper',
        ['How To Use?',
        'Extract Text',
        'Get Summary',
        'Question Answering'],
        icons=[':milky_way:',':moyai:',':moyai:',':moyai:']
        )

if(selected=='How To Use?'):
    st.title('Introduction')
    st.subheader('Problem Statement')
    st.write('''
    Despite advancements in technology the manual process of scanning and summarizing large
    volumes of documents remains time-consuming and error-prone. As a result, organizations and
    individuals are facing challenges in efficiently processing, organizing and comprehending the
    vast amounts of information contained in these documents.
    Hence the development of an automatic document scanner and summarizer system that can
    efficiently and accurately process large volumes of documents, extract key information, and
    present it in a concise, easy-to-understand format and provide the facility of question and answer
    will benefit students by saving time, reducing errors and improving the efficiency of their
    document management processes
    ''')
    st.subheader('Project Overview')
    st.write('''
      With the vast syllabus for all streams of studies there is a need to make studying more efficient.
Exam Helper is a project aimed at developing a system that can scan and process physical
documents and convert them into digital format, while also summarizing the key information
contained in the document and giving the ability for questioning and answering.
The system will contain:''')
    st.write('''1.Document Scanner: A component that would scan the physical document and convert it
into a digital image. This digital image undergoes a series of refinements and
improvements to ensure that the final result is of high quality and accurately reflects the
original document. The refinements may include correction of skew, removal of shadows
and blemishes, and enhancing the overall clarity and readability of the image.
''')
    st.write('''2.File conversion: It is an essential component that facilitates the process of digitizing
physical documents. The end result is a crisp, clear, and accurate representation of the
original document in a digital format (PDF format), making it easily accessible and
retrievable for future use.''')
    st.write('''3.Text extraction and Summarization: On the improved images, sophisticated OCR
algorithms will be implemented to ensure that text is extracted accurately and effectively.
After the text is extracted, industrial level algorithms will be applied to provide a
summary to the user which will provide an extensive overview which will be short
enough to save time but long enough to include all the important points as in the original
document.''')
    st.write('''4.Question and Answering system: After providing a summary of the document to the
user, a user friendly GUI will be provided to the user in which the user would be able to
ask questions related to the document which will be answered by our system.This makes
it easier for users to quickly find the information they need, without having to sift through the entire document. The GUI is intuitive and easy to use, allowing users to effortlessly
interact with the system and obtain the information they require. With its combination of
advanced technology and user-friendly design, this Question and Answering system will
be an indispensable tool for anyone who needs quick and easy access to information
contained in documents
''')
    st.subheader('Objectives')
    st.write('''1.To develop a system that can accurately and efficiently scan physical documents and convert them into digital format.
    ''')
    st.write('''2.To accurately recognize and convert the text contained in the scanned document into machine-readable format using optical character recognition (OCR) technology.
    ''')
    st.write(''' 3.To analyze the text contained in the document and generate a summary of the key information contained in the document using natural language processing and machine learning algorithms.
    ''')
    st.write('''4.To generate a user-friendly interface for performing the questioning and answering based on the document and displaying the summery.
    ''')

if(selected=='Extract Text'):
    uploaded_files = st.file_uploader('Upload your files', accept_multiple_files=True)
    reader = easyocr.Reader(['en'])
    if st.button('Extract'):
        sorted_files = sorted(uploaded_files, key=lambda file: file.name)

        extracted_texts = []
        for uploaded_file in sorted_files:
            image = Image.open(uploaded_file)
            results = reader.readtext(image, detail=0)
            extracted_texts.append(results)

        st.subheader('Extracted Text')
        for i, text in enumerate(extracted_texts):
            st.subheader(f'Extracted Text from Image {i+1}')
            st.write(text)

    # Convert the extracted texts to a single string
        all_text = " ".join(" ".join(text) for text in extracted_texts)

    # Save the extracted text to a .txt file
        with open("extracted_text.txt", "w", encoding="utf-8") as file:
            file.write(all_text)

        st.success("Text extracted successfully.")


    if st.button('Download Extracted Text'):
        if os.path.exists("extracted_text.txt"):
            with open("extracted_text.txt", "r", encoding="utf-8") as file:
                text = file.read()
            st.download_button(
            label="Download Extracted Text",
            data=text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )
        else:
            st.warning("No extracted text file found. Please extract text first.")

if(selected=='Get Summary'):
    uploaded_file = st.file_uploader('Upload Text File', accept_multiple_files=False)
    if(st.button('Submit')):
        data = uploaded_file.read().decode("utf-8")
        stopwords = list(STOP_WORDS)
        nlp = spacy.load('en_core_web_lg')
        doc = nlp(data)
        tokens = [token.text for token in doc]
        punctuation = punctuation + '\n'
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency

        sentence_tokens = [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        from heapq import nlargest
        select_length = int(len(sentence_tokens) * 0.3)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

        final_summary = [word.text for word in summary]
        summary = ' '.join(final_summary)
        with open("summary.txt", "w", encoding="utf-8") as file:
            file.write(summary)

        st.title('Summary')
        st.write(summary)


if(selected=='Question Answering'):
    # tokenizer = LongformerTokenizerFast.from_pretrained(
    # 'allenai/longformer-base-4096')
    tokenizer = LongformerTokenizer.from_pretrained(
    "valhalla/longformer-base-4096-finetuned-squadv1")
    # model = LongformerForQuestionAnswering.from_pretrained(
    # "valhalla/longformer-base-4096-finetuned-squadv1")
    context_tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    model = LongformerForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

    text=st.text_area('Context',height=350)
    question = st.text_input('Questions?')
    if(st.button('Submit')):
        def longformer(text, question):
            encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            start_scores, end_scores = model(input_ids, attention_mask=attention_mask,return_dict=False)
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            answer_tokens = all_tokens[torch.argmax(start_scores):torch.argmax(end_scores)+1]
            answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
            return answer
        ans= (longformer(text, question))
        st.subheader('Question-')
        st.write('Q.' + question)
        st.subheader("Answer:")
        st.write(ans)
