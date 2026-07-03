from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import  CharacterTextSplitter

loader = PyPDFLoader('Introduction to Docker.pdf')
docs = loader.load()

splitter  = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 3,
    separator=" "
)

result = splitter.split_documents(docs)
print(result[2].page_content)