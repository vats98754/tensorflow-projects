from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

sentences = [
	'I love my dog',
	'I love my cat',
	'You love my dog!',
	'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')  # id's of the 100 most frequent words used
tokenizer.fit_on_texts(sentences)  # tokenizes each word in each sentence; greater frequency => lower token
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post', truncating='pre', maxlen=5)  # maxlen is the length of the sentence
# with the most words by default, if the set value is less than that, shorter words are truncated to the specified
# length, with the post truncated by default. The padding of 0s can also be placed after the tokens using padding='post'

print(word_index)
print(sequences)
print(padded)

test_data = [
	'i really love my dog',
	'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)  # Should use OOV / Out-Of-Vocabulary token as the number 1
