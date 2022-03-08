import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

def grab_first(doc, words):
    words_left = words
    # Break document into sentences
    sentences = nltk.tokenize.sent_tokenize(doc)

    # Grab the number of sentences that are less than the number of words
    result_sentences = []
    for sentence in sentences:
        sent_len = len(sentence.split())
        if sent_len <= words_left:
            result_sentences.append(sentence)
            words_left -= sent_len
        else:
            if sent_len < 2 * words_left:
                result_sentences.append(sentence)
                break
    # Join the sentences together
    summary = ' '.join(result_sentences)
    return summary

print(grab_first("Hi I'm Daniel. Daniel likes to swim. I like to climb trees.", 4))
