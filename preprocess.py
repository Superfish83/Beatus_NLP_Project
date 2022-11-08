import io
from logging import PercentStyle
import tensorflow as tf
from sklearn.model_selection import *

def preprocess_sentence(text):
    text.strip()
    text = '<start> ' + text + ' <end>'

    return text

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    #텍스트 파일 불러오기
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')][1:]  for l in lines[:num_examples]]

    return zip(*word_pairs)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    # 전처리된 타겟 문장과 입력 문장 쌍을 생성합니다.
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))




#이하는 테스트

#dataset = create_dataset('sae4k-master/data/sae4k_v2.txt', 100)

#for pair in dataset:
    #print(pair)