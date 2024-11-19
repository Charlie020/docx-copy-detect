import hashlib
import os

from datetime import datetime
from docx import Document
from utils import get_doc_content, get_token_set_from_content


def store_docx_to_es(es, index_name, file_path):
    """
    Store or update a single doc into es.
    """
    if file_path.endswith('.docx'):
        doc = Document(file_path)
        content = get_doc_content(file_path)
        token_list = list(get_token_set_from_content(content))
        document = {
            "title": file_path[2:],
            "author": doc.core_properties.author if doc.core_properties.author is not None else 'Unknown',
            "content": content,
            "last_modified": doc.core_properties.modified.date() if doc.core_properties.modified is not None else 'Unknown',
            "stored_date": datetime.now().date(),
            "tokens": token_list
        }

        doc_id = hashlib.md5(document["title"].encode('utf-8')).hexdigest()
        if not es.exists(index=index_name, id=doc_id):
            es.index(index=index_name, id=doc_id,document=document)
            print(f"document {file_path[2:]} stored with ID {doc_id}")
        else:
            es.update(index=index_name, id=doc_id, body={"doc": document})
            print(f"document {file_path[2:]} updated with ID {doc_id}")
    else:
        print(f"'{file_path}': Cannot process .doc files, only .docx files are supported.")


def store_directory(es, index_name, directory_path):
    """
    Recursively store docs in the directory into es.
    """
    dirs = os.listdir(directory_path)
    if dirs:
        for dir in dirs:
            if os.path.isdir(os.path.join(directory_path, dir)):
                store_directory(es, index_name, os.path.join(directory_path, dir))
            elif os.path.isfile(os.path.join(directory_path, dir)):
                store_docx_to_es(es, index_name, os.path.join(directory_path, dir))


