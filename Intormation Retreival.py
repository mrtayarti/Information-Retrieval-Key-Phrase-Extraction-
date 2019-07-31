# encoding=utf-8#
import operator
import re
import string
import urllib
from itertools import chain, groupby
from urllib import request
import numpy as np
from bs4 import BeautifulSoup
from nltk import pos_tag_sents
from nltk import word_tokenize, sent_tokenize
from nltk.chunk import tree2conlltags
from nltk.chunk.regexp import RegexpParser
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# initialize variables
text_path = 'logs/text_log.txt'
plaintext_path = 'logs/plaintext_log.txt'
tf_idf_log_path = 'logs/words_score_log.txt'
sentence_path = 'logs/sentence_log.txt'
vocabs_word_path = 'logs/vocabs_word_log.txt'
vocabs_phrase_path = 'logs/vocabs_phrase_log.txt'
index_keyword_path = 'logs/top_keys_index.txt'
index_phrases_path = 'logs/top_phrases_index.txt'
pos_path = 'logs/pos_log.txt'
url_path = 'url.txt'
html_source, plain_text, list_plain_text = [], [], []
punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))
stop_words = set(stopwords.words('english'))
np.set_printoptions(threshold=np.inf)


# Step 1, get the request html
def get_html(url):
    req_html = request.urlopen(url)
    html = req_html.read().decode('utf-8').strip('\n')
    html_source.append(html)
    return html


def vocab_gen(texts, bool_key):
    list_word = []
    vocabs = []
    word_write = ""
    phrase_write = ""
    pos_write = ""
    sentences = sent_tokenize(texts)
    sentence_write = "\n".join(sentences)
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = list(map(lambda s: s.lower(), words))
        list_word.append(words)
    words_w_pos = pos_tag_sents(list_word)  # POS
    dumb = [j for sub in words_w_pos for j in sub]
    dumb = pos_tag_sents(dumb)
    dumb = [j for sub in dumb for j in sub]
    for i in dumb:
        pos_write += str(i)
        pos_write += "\n"
    # define grammar to pull out the phrases
    grammar = r'KT: ' \
              r'{' \
              r'(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+' \
              r'}'
    grammar = RegexpParser(grammar)
    all_tag = chain.from_iterable([tree2conlltags(grammar.parse(tag)) for tag in words_w_pos])
    for key, group in groupby(all_tag, lambda tag: tag[2] != 'O'):
        vocabs_temp = ' '.join([word for (word, pos, chunk) in group])
        if bool_key == 'Phrase':
            if key is True and vocabs_temp not in stop_words and len(vocabs_temp) > 2 and (' ' in vocabs_temp) == True:
                vocabs.append(vocabs_temp)
                phrase_write += vocabs_temp
                phrase_write += "\n"
        else:
            if key is True and vocabs_temp not in stop_words and len(vocabs_temp) > 2 and (' ' in vocabs_temp) == False:
                vocabs.append(vocabs_temp)
                word_write += vocabs_temp
                word_write += "\n"
    update_file = open(vocabs_word_path, 'w')
    update_file.write(word_write)
    if bool_key == 'Phrase':
        update_file = open(vocabs_phrase_path, 'w')
        update_file.write(phrase_write)
    update_file = open(sentence_path, 'w')
    update_file.write(sentence_write)
    update_file = open(pos_path, 'w')
    update_file.write(pos_write)
    return vocabs


def tf_idf_model(texts, corpus_num, choice, bool_key):
    vocabulary = [vocab_gen(unidecode(text), bool_key) for text in texts]
    if choice == '0':
        print("\n===========================================\n"
              "Possible indexing from merging", corpus_num,
              "corpus = ", len(vocabulary[0]), "set(s) (", vocabs_word_path, ")",
              "\n===========================================")
    else:
        print("\n===========================================\n"
              "Possible indexing from link number", choice, " =",
              len(vocabulary[0]), "set(s) (", vocabs_word_path, ")",
              "\n===========================================")
    vocabulary = list(chain(*vocabulary))
    vocabulary = list(np.unique(vocabulary))  # unique vocab

    max_vocab_len = max(map(lambda s: len(s.split(' ')), vocabulary))
    model = TfidfVectorizer(vocabulary=vocabulary, lowercase=True,
                            ngram_range=(1, max_vocab_len), stop_words=None,
                            min_df=0.1, max_df=0.7)
    X = model.fit_transform(texts)
    vocabulary_sort = [v[0] for v in sorted(model.vocabulary_.items(),
                                            key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(X.toarray()))

    # return list of top candidate phrase
    key_phrases = list()
    for sorted_array_doc in sorted_array:
        key_phrase = [vocabulary_sort[e] for e in sorted_array_doc[0:10]]
        key_phrases.append(key_phrase)

    return key_phrases


def get_text(url, corpus_num, loop_round):
    links = []
    source = urllib.request.urlopen(url).read().decode("utf-8")
    soup = BeautifulSoup(source, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    # Get Plain Text
    text = (soup.get_text())
    update_word = open(text_path, 'w')
    update_word.write(text)
    # Remove Spaces
    lines = (line.strip() for line in text.splitlines())
    # Breaking headlines
    chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
    # Remove blank lines
    text = ' '.join(chunk for chunk in chunks if chunk)
    list_plain_text.append(text)
    for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
        links.append(link.get('href'))
    print("[", loop_round, "/", corpus_num, "]\nExtracting from :", url)
    print("Total numbers of links found : ", len(links), "link(s)")

    # extracting images
    images = []
    for img in soup.findAll('img'):
        images.append(img.get('src'))
    print("Total numbers of images links found in : ", len(images),
          "link(s)\n===========================================")
    return text


def tf_idf(words):
    docs = [words]
    weight_lgs = ""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()

    for doc in range(len(weight)):
        data_map = {}
        for word_weight in range(len(word)):
            data_map[word[word_weight]] = weight[doc][word_weight]
    sorted_map = sorted(data_map.items(), key=lambda item: item[1], reverse=True)
    for v in sorted_map:
        weight_lgs += str(v)
        weight_lgs += '\n'
    return weight_lgs


def read_file(path):  # read the file and split it into list
    with open(path, "r") as file:
        return file.read()


def read_file_re(path):  # read the file and split it into list divided by a space
    with open(path, "r") as file:
        return [re.split("\s+", line.rstrip('\n')) for line in file]


# Calling function and displaying the result
def ini():
    list_url = read_file_re(url_path)
    update_file = open(plaintext_path, 'w')
    counter = 0
    for _ in list_url:
        str_url = ''.join(list_url[counter])
        plain_text.append(get_text(str_url, len(list_url), counter + 1))
        counter += 1
    selection = input(
        "\n=============[[[[[[MENU]]]]]]==============\nWhich link do you want to extract keywords & key phrases?\n"
        "Enter link number or '0' to merge plaintext from every links\n"
        ":")
    index_phrases = "Top 10 best key phrases\n\n"
    index_words = "Top 10 best key words\n\n"
    counter = 0
    if selection != '0':
        key_phrases = tf_idf_model([plain_text[int(selection) - 1]], len(list_url), selection, "Phrase")
        update_file.write(plain_text[int(selection) - 1])
        print("Top 10 key phrases (", index_phrases_path, ")")
        for _ in range(10):
            index_phrases += key_phrases[0][counter]
            index_phrases += '\n'
            print(counter + 1, ".", key_phrases[0][counter])
            counter += 1
        key_phrases = tf_idf_model([plain_text[int(selection) - 1]], len(list_url), selection, "Word")
        print("Top 10 key words (", index_keyword_path, ")")
        counter = 0
        for _ in range(10):
            index_words += key_phrases[0][counter]
            index_words += '\n'
            print(counter + 1, ".", key_phrases[0][counter])
            counter += 1
    elif selection == '0':
        merge_text = '\n'.join(list_plain_text)
        key_phrases = tf_idf_model([merge_text], len(list_url), selection, "Phrase")
        update_file.write(merge_text)
        print("Top 10 key phrases (", index_phrases_path, ")")
        for _ in range(10):
            index_phrases += key_phrases[0][counter]
            index_phrases += '\n'
            print(counter + 1, ".", key_phrases[0][counter])
            counter += 1
        key_phrases = tf_idf_model([merge_text], len(list_url), selection, "Word")
        print("Top 10 key words (", index_keyword_path, ")")
        counter = 0
        for _ in range(10):
            index_words += key_phrases[0][counter]
            index_words += '\n'
            print(counter + 1, ".", key_phrases[0][counter])
            counter += 1
    word_weight = tf_idf(read_file(vocabs_word_path))

    update_index_phrase = open(index_phrases_path, 'w')
    update_index_phrase.write(index_phrases)

    update_index_word = open(index_keyword_path, 'w')
    update_index_word.write(index_words)

    update_word_weight = open(tf_idf_log_path, 'w')
    update_word_weight.write(word_weight)
    print("\n[[=>------]] Text log has been updated (", text_path, ")")
    print("\n[[==>-----]] Plaintext log has been updated (", plaintext_path, ")")
    print("\n[[===>----]] Sentence log has been updated (", sentence_path, ")")
    print("\n[[====>---]] POS tagger log has been updated (", pos_path, ")")
    print("\n[[======>-]] Dictionary weights log has been updated (", tf_idf_log_path, ")")
    print("\n[[=======>]] NER tagger log has been updated (", pos_path, ")")


ini()
