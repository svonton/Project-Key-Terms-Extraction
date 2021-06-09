import pandas as pd
from lxml import etree
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option("display.max_rows", None, "display.max_columns", None)
lemmatizer = WordNetLemmatizer()
data = etree.parse('news.xml').getroot()
trash_can = stopwords.words('english') + list(punctuation)
all_news = [news[1].text.replace('\n', ' ') for news in data[0]]
ft_dataset = list()
ft_data_header = list()

for news in data[0]:
    tokenized_words = word_tokenize(news[1].text.lower())
    lem_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
    for trash in trash_can:
        while trash in lem_words:
            lem_words.remove(trash)
    lem_words = [pos_tag([word])[0][0] for word in lem_words if pos_tag([word])[0][1] == 'NN']
    ft_data_header.append(news[0].text)
    ft_dataset.append(' '.join(lem for lem in lem_words))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(ft_dataset)
terms = vectorizer.get_feature_names()

for i in range(len(ft_data_header)):
    print(ft_data_header[i] + ':')
    df = pd.DataFrame(tfidf_matrix[i].toarray())
    df = df.transpose().sort_values(by=0, ascending=False).reset_index()
    string_to_print = {terms[int(df.iloc[j]['index'])]: df.iloc[j][0] for j in range(len(ft_data_header))}
    most_common = (sorted(string_to_print.items(), key=lambda t: (t[1], t[0]), reverse=True))
    top_5 = [k for k, v in most_common[:5]]
    print(' '.join(i for i in top_5)+'\n')