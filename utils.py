import re
import hanlp
import math

from elasticsearch import Elasticsearch
from docx import Document


def connect_elasticsearch(hosts='http://127.0.0.1:1200', username='elastic', password='infini_rag_flow'):
    es = Elasticsearch(
        hosts=[hosts],
        basic_auth=(username, password),
        request_timeout=30
    )

    assert es.ping(), "Connection failed!"

    print("Connection established!")
    return es

def get_doc_content(file_path):
    """
    Get content in .docx, including text, table.
    """
    doc = Document(file_path)
    content = ""
    cnt = 0  # 记录当前遍历到的表格数
    for element in doc.element.body:
        if element.tag.endswith("p"):  # 段落元素
            paragraph = element.xpath(".//w:t")
            if paragraph:
                text = ''.join([node.text for node in paragraph if node.text])
                content += text + "\n"
        elif element.tag.endswith("tbl"):  # 表格元素
            for row in doc.tables[cnt].rows:
                row_text = '\t'.join(cell.text.strip() for cell in row.cells if cell.text)
                content += row_text + "\n"
            cnt += 1
    return content

def extract_sentences(content):
    res = re.sub(r' ', '', content)
    res = re.sub(r'\n+', ' ', res)
    res = re.sub(r'\t+', ' ', res)
    res = re.sub(r'。+', '。 ', res)
    res = re.sub(r'！+', '！ ', res)
    res = re.sub(r'？+', '？ ', res)
    res = re.sub(r' +', ' ', res)

    return [sentence for sentence in res.split(' ') if len(sentence) >= 13]

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = {line.strip() for line in f if line.strip()}  # 去掉多余的空行
    return stopwords

def hanlp_tokenizer(sentences, stopwords='./stopwords.txt'):
    """
    Get tokens for each sentence
    """
    tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    tokens_list = tokenizer(sentences)
    stopwords = load_stopwords(stopwords)
    return [[token for token in tokens if token not in stopwords] for tokens in tokens_list]

def get_token_set_from_content(content, stopwords_path='./stopwords.txt'):
    sentences = extract_sentences(content)
    tokens_list = hanlp_tokenizer(sentences, stopwords=stopwords_path)
    token_set = set()
    for tokens in tokens_list:
        for token in tokens:
            token_set.add(token)
    return token_set

def compute_tf(tokens_list):
    token_tf = {}
    for tokens in tokens_list:
        for token in tokens:
            token_tf[token] = token_tf.get(token, 0) + 1
    return token_tf

def compute_idf(tokens_set, docs):
    token_idf = {}
    for token in tokens_set:
        for doc in docs:
            if token in doc['_source']['tokens']:
                token_idf[token] = token_idf.get(token, 0) + 1
    for token, idf in token_idf.items():
        token_idf[token] = math.log(1.0 * len(docs) / (1 + idf))
    return token_idf

def get_keywords_base_on_tfidf(file_path, es, index_name):
    content = get_doc_content(file_path)

    sentences = extract_sentences(content)
    tokens_lists = hanlp_tokenizer(sentences)
    tf = compute_tf(tokens_lists)

    tokens_set = get_token_set_from_content(content)
    response = es.search(index=index_name)
    docs = response['hits']['hits']
    idf = compute_idf(tokens_set, docs)

    tfidf = {}
    for token in tokens_set:
        tfidf[token] = tf[token] * idf[token]

    sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    top_tfidf_words = []
    top_value = sorted_tfidf[0][1] if sorted_tfidf else 0
    top_values = [top_value]
    top_value_cnt = 1

    for token, value in sorted_tfidf:
        if value == top_value:
            top_tfidf_words.append(token)
        elif top_value_cnt < 10:
            top_tfidf_words.append(token)
            top_value = value
            top_values.append(value)
            top_value_cnt += 1
        if top_value_cnt == 10:
            break

    return top_tfidf_words, top_values

def search_documents(es, index_name, keyword="", field="content", size=10):
    if isinstance(keyword, str) and len(keyword) != 0:
        query = {
            "size": size,
            "query": {
                "match": {
                    field: keyword
                }
            }
        }
    elif isinstance(keyword, list):
        query = {
            "size": size,
            "query": {
                "bool": {
                    "should": [
                        {"match": {field: k}} for k in keyword
                    ],
                    "minimum_should_match": 1  # 至少匹配一个关键字
                }
            }
        }
    else:
        query = {
            "size": size,
            "query": {
                "match_all": {}
            }
        }

    response = es.search(index=index_name, body=query)
    docs = response['hits']['hits']

    if docs:
        print(f"Index name: {index_name}, number of docs returned is {len(docs)}, using query keyword: '{keyword}':")
        for i, doc in enumerate(docs):
            print(f"\tdocument {i + 1}: {doc['_score']}, {doc['_source']['title']}, {doc['_id']}, {doc['_source']['author']}, {doc['_source']['last_modified']}, {doc['_source']['stored_date']}")
        return docs
    else:
        print(f"No documents found in '{index_name}', using query keyword: '{keyword}':")
    print("\n")


