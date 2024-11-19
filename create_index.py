import os


def create_index(es, index_name, mappings=None):
    """
    creates an index in es
    """
    if mappings is None:
        mappings = {
            "properties": {
                "title": {"type": "text"},
                "author": {"type": "keyword"},
                "content": {"type": "text"},
                "last_modified:": {"type": "date"},
                "stored_date": {"type": "date"},
                "tokens": {"type": "keyword"}
            }
        }
    if not es.indices.exists(index=index_name):
        try:
            es.indices.create(index=index_name, body={"mappings": mappings})
            print(f"Index '{index_name}' created successfully!")
        except Exception as e:
            print(f"Error creating index '{index_name}': {e}")
    else:
        print(f"Index '{index_name}' already exists.")
    es.indices.create(index="5、运维服务内审、管审", body={"mappings": mappings})

def create_index_based_on_folder_name(es, directory_path, mappings=None):
    """
    Create index in es based on the folder name in the directory
    """
    if mappings is None:
        mappings = {
            "properties": {
                "title": {"type": "text"},
                "author": {"type": "keyword"},
                "content": {"type": "text"},
                "last_modified:": {"type": "date"},
                "stored_date": {"type": "date"}
            }
        }
    dir13 = os.listdir(directory_path)
    for dir_name in dir13:
        dir_path = os.path.join(directory_path, dir_name)
        if os.path.isdir(dir_path):
            create_index(es, dir_name, mappings)