import streamlit as st
import openai, boto3
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="PostGame Extract", page_icon=":tada:", layout="wide")
st.title("Our Project's aim is to make alleviate the hassle of manual transcription and hard to read, handwritten documents in the sports world")

file_upload = st.sidebar.file_uploader('Please Upload a File')

image = Image.open('menu.jpeg')
st.sidebar.image(image, caption='menu',use_column_width=True)

textract = boto3.client('textract', region_name='us-east-1', aws_access_key_id='AKIASF2P7IDFKIGWFHIU',aws_secret_access_key='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')

def document_to_retriever(document, chunk_size, chunk_overlap):
    with open(document, 'rb') as file:
        img = file.read()
        bytes = bytearray(img)

    extracted_text = textract.analyze_document(Document = {'Bytes': bytes}, FeatureTypes = ['TABLES'])


    text = []
    blocks = extracted_text['Blocks']

    for block in blocks:
        if block['BlockType'] == 'WORD':
             text.append(block['Text'])
    # text formation based upon Line block type
        elif block['BlockType'] == 'LINE':
             text.append(block['Text'])

    words = " ".join(text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.create_documents([words])
    print(splits)
    vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings(openai_api_key='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'))
    retriever = vectorstore.as_retriever()


    return retriever

retriever = document_to_retriever('menu.jpeg', 100,2)

def generate_response(retriever, input_text):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-4", temperature=0,
                                                          openai_api_key='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'),
                                               retriever=retriever, verbose=True, memory=memory)
    st.info(qa({"question":input_text}))



with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(retriever,text)


