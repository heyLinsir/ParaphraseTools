import random
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
print('Load %d stop words.' % (len(stop_words)))

def remove_stop_words(words, save_prob=0):
    return [word for word in words if word not in stop_words or random.random() < save_prob or word.lower() in ['i', 'he', 'she', 'it', 'me', 'him', 'her', 'us', 'our', 'we', 'my', 'his']]

def shuffle(words, shuffle_prob=0):
    if random.random() < shuffle_prob:
        random.shuffle(words)
    return words

def get_synonym_words(word):
    word_set = wn.synsets(word)
    syn_set = [word.lemma_names()[0] for word in word_set]
    syn_set = list(set(syn_set))
    return syn_set

def replace_synonym_words(words, replace_prob=0):
    new_words = []
    for word in words:
        if random.random() < replace_prob and word.lower() not in ['i', 'he', 'she', 'it', 'me', 'him', 'her', 'us', 'our', 'we', 'my', 'his']:
            syn_set = get_synonym_words(word)
            if syn_set != []:
                new_words.append(random.choice(syn_set))
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return new_words

def corrupt(sentence, save_prob=.5, shuffle_prob=.2, replace_prob=.5):
    words = word_tokenize(sentence.strip().lower())

    words = remove_stop_words(words, save_prob)
    words = shuffle(words, shuffle_prob)
    words = replace_synonym_words(words, replace_prob)

    return ' '.join(words)

if __name__ == '__main__':
    print(corrupt('How do you send a private message to someone you’re following on quora?', save_prob=.5, shuffle_prob=.2, replace_prob=.8))