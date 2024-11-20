from copy_detection import copy_detect
from utils import connect_elasticsearch, search_documents


def show_indices_in_es(es):
    indices_info = es.cat.indices(format="json")
    print("All indices:")
    for i, index_info in enumerate(indices_info):
        print(f"{i + 1}:\tIndex name: ‘{index_info['index']}’, number of docs: {index_info['docs.count']}")
    print("\n")

def get_doc_content_by_id(es, index_name, doc_id):
    response = es.get(index=index_name, id=doc_id)
    return response['_source']['content']

def delete_documents(es, index_name, doc_id):
    es.delete(index=index_name, id=doc_id)
    print(f"{index_name}'s docs after delete:")
    search_documents(es, index_name)

if __name__ == '__main__':
    es = connect_elasticsearch()

    input_file_path = './1.docx'
    copy_detect(input_file_path, es, '5、运维服务内审、管审', choice='text', num_chunks=1)
    pass

