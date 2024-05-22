import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import openai
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import re
import random
import os

# load_dotenv(find_dotenv())

from get_embedding_function import get_embedding_function

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Generate a random multiple choice question with four choices from the context:

{context}

---

Provide the correct answer with the format Correct Answer:.
"""

EVAL_PROMPT = """
Provide the correct answer to this question:
{expected_response}
---
Compare the correct answer above with this response:
{actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
---
What is the correct answer to the first question above.
"""


def main():
    # ### QA type
    # # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text
    # query_rag(query_text)

    st.set_page_config(page_title="Australian Citizenship - Knowledge tester")
    st.header("Anything about Australian citizenship - Reviewer ")
    ### Questionnaire type
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    if 'question' not in st.session_state:
        st.session_state.question = ""
    if 'answer' not in st.session_state:
        st.session_state.answer = ""

    # if st.session_state.stage == 0:
    #     st.button('Generate question', on_click=set_state, args=[1])
    #     st.write(st.session_state.stage)

    # if st.session_state.stage == 1:
    #     if st.session_state.answer == "":
    #         query_text = "Generate a random multiple choice question and provide the correct answer"
    #         question = query_rag(query_text)
    #         st.write(question)
    #         st.session_state.question = question
    #         response = st.text_input("What is your answer:", key="answer")
    #     else:
    #         st.session_state.stage = 2

    # if st.session_state.stage == 2:
    #     # st.write(st.session_state.question)
    #     question = st.session_state.question
    #     # response = st.session_state.answer
    #     st.write(question)
    #     # st.write("Your answer is: ", response)
    #     response = st.text_input("What is the answer:", key="answer")
    #     if response:
    #         prompt, answer = answer_rag(question, response)
    #         st.write(prompt)
    #         st.write(answer)
    #         st.button("Generate next question", on_click=clear_text)
    if 'button' not in st.session_state:
        st.session_state.button = False
    if st.session_state.button == False:
        query_text = "Generate a random multiple choice question and provide the correct answer"
        question = query_rag(query_text, os.environ['OPENAI_API_KEY'])
    else:
        question = st.session_state.question, st.session_state.answer
    st.write(question[0])
    st.session_state.question = question[0]
    try:
        st.session_state.answer = question[1]
        st.button("Show Answer", on_click=click_button, disabled=st.session_state.button)

        if st.session_state.button == True:
            # The message and nested widget will remain on the page
            # st.write(st.session_state.question)
            st.write("Correct Answer: ", st.session_state.answer)
            st.button("New question", on_click=clear_text)
    except IndexError:
        st.write('Please try again. There is an error. 🙁')
        st.button("New question", on_click=clear_text)




def click_button():
    st.session_state.button = True

# def set_state(i):
#     st.session_state.stage = i

def clear_text():
    st.session_state.button = False
    st.session_state.answer = ""
    st.session_state.question = ""


def query_rag(question: str, key: str):
    # ### QA type
    # # Prepare the DB.
    # embedding_function = get_embedding_function()
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # # Search the DB.
    # results = db.similarity_search_with_score(question, k=5)

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=question)
    # # print(prompt)

    # # model = Ollama(model="mistral")
    # # response_text = model.invoke(prompt)

    # model = ChatOpenAI(model_name= "gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY)
    # response_text = model.invoke(prompt)

    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text.content}\n\nSources: {sources}"

    ### Questionnaire type
    # Prepare the DB.
    embedding_function = get_embedding_function(key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Search the DB.
    results = db.similarity_search_with_score(question, k=5)
    rand_chunk = random.randint(0, len(db.get()['ids'])-1)
    results = db.get(ids=db.get()['ids'][rand_chunk])['documents']

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    question_prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    question_prompt = question_prompt_template.format(context=results[0], question=question)
    # print(prompt)

    # model = Ollama(model="mistral")
    # response_text = model.invoke(prompt)

    question_model = ChatOpenAI(api_key=key)
    question_text = question_model.invoke(question_prompt)

    # sources = results.metadata.get("id", None)
    # print(sources)
    # question_output = question_text.content.split('Correct Answer:')
    question_output = re.split("correct answer: ", question_text.content, flags=re.IGNORECASE)
    # formatted_response = f"Response: {question_text.content}\n\nSources: {sources}" 
    question_output[0] = question_output[0].replace("\n", "  \n")
    print("Sample: " + str(question_text))
    print("Sample123: " + str(question_output))
    print("Sample12345: " + str(question_output[0]))

    return question_output

# def answer_rag(question: str, response: str):

#     # prompt_template = ChatPromptTemplate.from_template(EVAL_PROMPT)
#     prompt = EVAL_PROMPT.format(expected_response=question, actual_response=response)

#     response_model = ChatOpenAI(model_name= "gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY)
#     evaluation_results_str = response_model.invoke(prompt)
#     evaluation_results_str_cleaned = evaluation_results_str.content.strip().lower()

#     print(prompt)

#     if "true" in evaluation_results_str_cleaned:
#         # Print response in Green if it is correct.
#         print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
#         outcome = "Your answer is correct"
#         return evaluation_results_str.content, outcome
#     elif "false" in evaluation_results_str_cleaned:
#         # Print response in Red if it is incorrect.
#         outcome = print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
#         outcome = "Your answer is incorrect "
#         return evaluation_results_str.content, outcome
#     # else:
#     #     raise ValueError(
#     #         outcome = f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
#     #     )
#     #     outcome = f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
#     #     return evaluation_results_str.content, outcome


# demo = gradio.Interface(fn=query_rag, inputs = "text", outputs = "text", title = "Anything about Australian Citizenship - Ask me anything")
# demo.launch(share=True)




if __name__ == "__main__":
    main()
