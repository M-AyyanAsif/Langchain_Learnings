from langchain_community.document_loaders import WebBaseLoader
url  = "https://lms.digiskills.pk/Default.aspx?id=2b8a5e25-67f9-4683-958e-04085fa51cb0"
loader = WebBaseLoader(url)
docs = loader.load()

print(len(docs))
print(docs[0].page_content)