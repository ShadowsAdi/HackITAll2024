from langchain import hub
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
# Set up RetrievelQA model
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")

#load the LLM
def load_llm():
    llm = Ollama(
    base_url="http://4.231.234.232:11434",
    model="mistral",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    # 4 threads for speed
    num_thread=4,
    # high repeat penalty to not repeat itself
    repeat_penalty=1.0,
    top_k=20,
    top_p=0.6,
    )
    return llm

def retrieval_qa_chain(llm,vectorstore):
 qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={'k': 6}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
)
 return qa_chain

def qa_bot(): 
    llm=load_llm() 
    DB_PATH = "vectorstores/db/"
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings())

    qa = retrieval_qa_chain(llm,vectorstore)
    return qa 

@cl.on_chat_start
async def start():
    chain=qa_bot()
    msg=cl.Message(content="Pornire chatbot...")
    await msg.send()
    msg.content= "Salut, cu ce te pot ajuta azi?"
    await msg.update()
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain=cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
    stream_final_answer=True,
    answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    cb.answer_reached=False
    res=await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")
    answer=res["result"]
    answer=answer.replace(".",".\n")
    sources=res["source_documents"]

    page = -1
    document = -1
    if sources:
        # retrieve only the last item from the list because it's the main source page
        page, document = split_document(sources[-1])
        print(f"Page: {page}")
        print(f"Source: {document}")

    if sources:
        answer+=f"\nSursa: " + str(document) + "\nPagina: " + str(page)
 
        for source in sources:
            page, document = split_document(source)
            answer+=f"\nAlte surse: " + str(document) + "\nPagina: " + str(page)
    else:
        answer+=f"\nNo Sources found"

    await cl.Message(content=answer).send() 

def split_document(document):
  page = document.metadata.get("page") + 1
  source = document.metadata.get("source")

  return page, source