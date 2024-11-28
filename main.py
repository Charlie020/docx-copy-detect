import argparse

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
    parser = argparse.ArgumentParser(description="Parameters for copy detect")
    parser.add_argument("--hosts", type=str, default="http://127.0.0.1:1200", required=False, help="Elasticsearch hosts")
    parser.add_argument("--username", type=str, default="elastic", required=False, help="Elasticsearch username")
    parser.add_argument("--passwd", type=str, default="infini_rag_flow", required=False, help="Elasticsearch password")
    parser.add_argument("--file", "-f", type=str, default="./1.docx", required=False, help="Path of the input file")
    parser.add_argument("--index", "-i", type=str, default="5、运维服务内审、管审", required=False, help="Index name")
    parser.add_argument("--choice", "-c", type=str, default="text", required=False, help="Type of the detection: semantic or text")
    parser.add_argument("--num_chunks", "-n", type=int, default=1, required=False, help="Number of the chunks")

    args = parser.parse_args()
    es = connect_elasticsearch(hosts=args.hosts, username=args.username, password=args.passwd)
    input_file_path = args.file
    copy_detect(input_file_path, es, args.index, choice=args.choice, num_chunks=args.num_chunks)

    pass

