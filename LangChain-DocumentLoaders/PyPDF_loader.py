from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader,UnstructuredPDFLoader,AmazonTextractPDFLoader,PyMuPDFLoader

loader = PyPDFLoader("Introduction to Docker.pdf")

docs = loader.load()
print(docs)

print(docs[5].metadata)
print(docs[5].page_content)

