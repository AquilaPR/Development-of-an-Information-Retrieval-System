from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit, QTextBrowser
from PyQt5.uic import loadUi
from rank_bm25 import BM25Okapi
import os
from PyPDF2 import PdfReader
from docx import Document
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import glob
from collections import Counter, OrderedDict

import nltk
nltk.download('punkt')



class Temubalik(QMainWindow):
    def __init__(self):
        super(Temubalik, self).__init__()
        loadUi('example.ui', self)

        self.pushButton.clicked.connect(self.search_query)


        self.textBrowsers = [self.textBrowser, self.textBrowser_2, self.textBrowser_3]


        self.stemmedWordsBrowsers = [self.textBrowser_5, self.textBrowser_6, self.textBrowser_7]
        self.allWordsBrowsers = [self.textBrowser, self.textBrowser_2, self.textBrowser_3]


        self.queryRelevanceBrowsers = [self.textBrowser_8, self.textBrowser_9, self.textBrowser_10]


        self.filesListBrowser = self.textBrowser_11

    def search_query(self):
        query = self.lineEdit.text()
        results = self.execute_search(query)


        self.textBrowser_4.clear()
        for stemmed_words_browser in self.stemmedWordsBrowsers:
            stemmed_words_browser.clear()
        for all_words_browser in self.allWordsBrowsers:
            all_words_browser.clear()
        for query_relevance_browser in self.queryRelevanceBrowsers:
            query_relevance_browser.clear()

        # Tampilkan hasil pada setiap textBrowser
        for i, (file_path, score) in enumerate(results):
            text_browser = self.textBrowsers[i]
            text_browser.clear()  # Hapus teks sebelumnya
            text_browser.append(f"Relevance score for file {file_path}: {score}\n\n")


            stemmed_words_info = f"Stemmed words and their counts:\n{display_word_counts(file_path, stem_all=True)}\n\n"
            if i < len(self.stemmedWordsBrowsers):
                stemmed_words_browser = self.stemmedWordsBrowsers[i]
                stemmed_words_browser.append(stemmed_words_info)


            all_words_info = f"All words in the file:\n{display_words(file_path, stem_all=False)}\n\n"
            if i < len(self.allWordsBrowsers):
                all_words_browser = self.allWordsBrowsers[i]
                all_words_browser.append(all_words_info)


            query_relevance_info = get_query_relevance_info(query, file_path, stem_all=True)
            query_relevance_text = f"Query relevance info in file {query_relevance_info}\n\n"
            if i < len(self.queryRelevanceBrowsers):
                query_relevance_browser = self.queryRelevanceBrowsers[i]
                query_relevance_browser.append(query_relevance_text)


        highest_score, summary_text = display_summary(results)
        self.textBrowser_4.append(summary_text)


        self.display_files_list()

    def execute_search(self, query):
        directory = 'c:/TemuBalik'
        files = get_files_in_directory(directory)
        corpus = [read_text_from_document(file) for file in files]
        tokenized_corpus, _ = zip(*[preprocess_text(doc, stem_all=True) for doc in corpus])

        bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
        tokenized_query, _ = preprocess_text(query, stem_all=True)
        results = []
        for file_index, file_path in enumerate(files):
            doc_scores = bm25.get_scores(tokenized_query)
            score = doc_scores[file_index]
            results.append((file_path, score))


        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        return sorted_results

    def display_files_list(self):
        directory = 'c:/TemuBalik'
        files_list = get_files_in_directory(directory)
        files_list_text = "List of Files in Directory:\n"
        for file_path in files_list:
            files_list_text += f"{file_path}\n"
        self.filesListBrowser.clear()
        self.filesListBrowser.append(files_list_text)


def read_text_from_document(file_path):
    _, file_extension = os.path.splitext(file_path.lower())

    if file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    elif file_extension == ".pdf":
        return read_text_from_pdf(file_path)
    elif file_extension == ".docx":
        return read_text_from_docx(file_path)
    else:
        return None


def read_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text


def read_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + " "
    return text


def preprocess_text(text, stem_all=False):
    stemmer = StemmerFactory().create_stemmer()
    tokens = word_tokenize(text.lower())

    if stem_all:
        processed_tokens = [stemmer.stem(token) if token.isalnum() else token for token in tokens]
    else:
        processed_tokens = [token for token in tokens if token.isalnum()]

    return tokens, processed_tokens


def get_files_in_directory(directory):
    return glob.glob(f"{directory}/*.*")


def display_words(file_path, stem_all=False):
    text = read_text_from_document(file_path)
    _, words = preprocess_text(text, stem_all=stem_all)
    return words


def display_word_counts(file_path, stem_all=False):
    _, processed_words = preprocess_text(read_text_from_document(file_path), stem_all=stem_all)
    word_counts = Counter(processed_words)
    return word_counts


def get_query_relevance_info(query, file_path, stem_all=True):
    _, processed_words_in_file = preprocess_text(read_text_from_document(file_path), stem_all=stem_all)
    query_relevance_info = OrderedDict()

    tokenized_query, _ = preprocess_text(query, stem_all=stem_all)
    if not any(token.isalnum() for token in tokenized_query):
        # Jika query tidak memiliki karakter alphanumerik, tambahkan query keseluruhan
        count = processed_words_in_file.count(query.lower())  # Ubah ke lowercase untuk keseragaman
        query_relevance_info[query.lower()] = count
    else:
        for query_word in sorted(set(tokenized_query)):
            if stem_all:
                stemmed_query_word = preprocess_text(query_word, stem_all=True)[1][0]
                count = processed_words_in_file.count(stemmed_query_word)
            else:
                count = processed_words_in_file.count(query_word)

            query_relevance_info[query_word] = count

    return query_relevance_info


def display_summary(sorted_results):
    highest_score_file, highest_score = sorted_results[0]

    summary_text = f"Summary:\n"
    summary_text += f"File with the highest relevance score is {highest_score_file} with a score of {highest_score}\n"

    return highest_score, summary_text


if __name__ == "__main__":
    app = QApplication([])
    window = Temubalik()
    window.show()
    app.exec_()