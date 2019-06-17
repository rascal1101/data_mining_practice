from gensim.models.word2vec import Word2Vec
import ast
import pandas as pd

import warnings

import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def create_model(filename, skip_gram=False):
    tokens = pd.read_csv(filename)
    tokens = tokens[tokens["contents"].apply(lambda x: 'http' not in x)]

    sentence = tokens["token"].apply(lambda x: ast.literal_eval(x)).tolist()

    if skip_gram:
        model = Word2Vec(sentence, min_count=50, iter=20, size=512, sg=1)
    else:
        model = Word2Vec(sentence, min_count=50, iter=20, size=512, sg=0)

    model.init_sims(replace=True)
    # model.save("./result/embedding.model")
    model.save("./result/" + model_name)


keywords = ["영화", "음악", "사랑", "배우"]
model_name = "test.model"
model_names = ["test.model", "test2.model", "test3.model"]
warnings.filterwarnings(action='ignore')


def most_similar():
    for m_name in model_names:
        model = Word2Vec.load("./result/" + m_name)

        print(model)
        for keyword in keywords:
            print(keyword + "와 관련된 키워드 : ", model.most_similar(keyword))

if __name__ == '__main__':
    # create_model("./result/all_token_1.csv")
    # create_model("./result/134963_all_token.csv")
    most_similar()

# word2vec 문서 : https://radimrehurek.com/gensim/models/word2vec.html
# tokenizing, word2vec 참고 : https://github.com/ssooni/data_mining_practice
# 참고한 블로그 : https://ssoonidev.tistory.com/93
# 데이터 참고 : https://github.com/lovit/soy/tree/master/data/naver_movie/comments
# 여기도 가능 : https://github.com/lovit/soynlp/tree/master/data