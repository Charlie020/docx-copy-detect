import hanlp
import multiprocessing
import math
multiprocessing.set_start_method('spawn', force=True)

from tqdm import tqdm
from utils import get_keywords_base_on_tfidf, search_documents, get_doc_content, extract_sentences, hanlp_tokenizer


def semantic_detect_for_chunk(sentences, related_docs_sentences, sts):
    result = []
    for i, sentence in tqdm(enumerate(sentences), desc="Processing sentence chunks..."):
        for j, related_sentences in enumerate(related_docs_sentences):
            compare_sentence_list = [(sentence, related_sentence) for related_sentence in related_sentences]
            sims = sts(compare_sentence_list)
            result.extend([(i, j, idx, sim) for idx, sim in enumerate(sims) if sim > 0.7])
    return result

def semantic_detect(sentences, related_docs_sentences, num_chunks=2):  # Cuda Memory = 2313 MB * num_chunks
    similar_sentences = []

    sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)

    chunk_size = len(sentences) // num_chunks
    chunks = [sentences[i:i + chunk_size] if i + chunk_size <= len(sentences) else sentences[i:len(sentences)] for i in
              range(0, len(sentences), chunk_size)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tasks = []
        for chunk in chunks:
            tasks.append(pool.apply_async(semantic_detect_for_chunk, (chunk, related_docs_sentences, sts)))

        for task in tqdm(tasks, desc="Processing sentences..."):
            result = task.get()
            similar_sentences.extend(result)

    return similar_sentences

# def semantic_detect(sentences, related_docs_sentences):
#     similar_sentences = []
#     sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)   # 2313 MB Cuda Memory
#     for i, sentence in enumerate(tqdm(sentences, desc="Processing sentences")):
#         for j, related_sentences in enumerate(related_docs_sentences):
#             compare_sentence_list = []
#             for related_sentence in related_sentences:
#                 compare_sentence_list.append((sentence, related_sentence))
#             for idx, sim in enumerate(sts(compare_sentence_list)):
#                 if sim > 0.6:
#                     similar_sentences.append((i, j, idx, sim))
#     return similar_sentences


def text_detect_for_chunk(sentences_tokens, related_docs_sentences_tokens):
    result = []
    for i, sentence in tqdm(enumerate(sentences_tokens), desc="Processing sentence chunks..."):
        for j, related_sentences_tokens in enumerate(related_docs_sentences_tokens):
            for k, related_sentence_tokens in enumerate(related_sentences_tokens):
                cnt = 0
                for token in sentence:
                    if token in related_sentence_tokens:
                        cnt += 1
                sim = 2.0 * cnt / (len(sentence) + len(related_sentence_tokens))
                if sim > 0.6:
                    result.extend([(i, j, k, sim)])
    return result

def text_detect(sentences, related_docs_sentences, num_chunks=2):
    similar_sentences = []

    sentences_tokens = hanlp_tokenizer(sentences)
    related_docs_sentences_tokens = []
    for related_sentences in related_docs_sentences:
        related_docs_sentences_tokens.append(hanlp_tokenizer(related_sentences))

    chunk_size = len(sentences_tokens) // num_chunks
    chunks = [sentences_tokens[i:i + chunk_size] if i + chunk_size <= len(sentences_tokens) else sentences_tokens[i:len(sentences_tokens)] for i in
              range(0, len(sentences_tokens), chunk_size)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tasks = []
        for chunk in chunks:
            tasks.append(pool.apply_async(text_detect_for_chunk, (chunk, related_docs_sentences_tokens)))

        for task in tqdm(tasks, desc="Processing sentences..."):
            result = task.get()
            similar_sentences.extend(result)
    
    return similar_sentences

def copy_detect(file_path, es, index_name, choice):
    valid_choices = ['semantic', 'text']
    if choice in valid_choices:
        content = get_doc_content(file_path)
        sentences = extract_sentences(content)

        keywords, _ = get_keywords_base_on_tfidf(file_path, es, index_name)
        related_docs = search_documents(es, index_name, keyword=keywords, field="tokens")
        related_docs_sentences = []
        for related_doc in related_docs:
            related_docs_sentences.append(extract_sentences(related_doc['_source']['content']))

        similar_sentences = []
        if choice == 'semantic':
            similar_sentences = semantic_detect(sentences, related_docs_sentences, num_chunks=1)
        elif choice == 'text':
            similar_sentences = text_detect(sentences, related_docs_sentences, num_chunks=1)
        print(similar_sentences)
    else:
        raise ValueError(f"Invalid choice '{choice}'. Must be one of {valid_choices}.")


