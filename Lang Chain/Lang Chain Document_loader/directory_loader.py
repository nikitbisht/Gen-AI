from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader,PyPDFDirectoryLoader, TextLoader, CSVLoader
import os

loader = DirectoryLoader(
    path='Books',
    glob='*.pdf', # give patten to select files
    loader_cls=PyPDFLoader
)
# loader = PyPDFDirectoryLoader('Books')
docs = loader.lazy_load()
for document in docs:
    print((document.metadata))

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

# for multiple loader 
def custom_loader(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
