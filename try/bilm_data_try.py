from bilm.data import Vocabulary, UnicodeCharsVocabulary

'''
UE for Vocabulary class
'''
print('\n\n\tUE for Vocabulary class:')
vocab_file = '../data/vocab_seg_chars_elmo.txt'
vocab_chars = Vocabulary(vocab_file, True)

print('====> bos: {}'.format(vocab_chars.bos))
print('====> eos: {}'.format(vocab_chars.eos))
print('====> unk: {}'.format(vocab_chars.unk))
print('====> size: {}'.format(vocab_chars.size))

word = '阿道夫'
print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))
word = '阿'
print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))

id = 234
print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))
id = 234234
print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))

word = '阿道夫'
print('====> encoded result without split: {}'.format(vocab_chars.encode(word)))
print('====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
word = '阿 道 夫'
print('====> encoded result without split: {}'.format(vocab_chars.encode(word)))
print('====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
print('====> encoded result with split with reverse: {}'.format(vocab_chars.encode(word, split=True, reverse=True)))

# input words
print('\n\n\n====> Words Vocabulary: ')
vocab_file = '../data/vocab_seg_words_elmo.txt'
vocab_chars = Vocabulary(vocab_file, True)
print('====> bos: {}'.format(vocab_chars.bos))
print('====> eos: {}'.format(vocab_chars.eos))
print('====> unk: {}'.format(vocab_chars.unk))
print('====> size: {}'.format(vocab_chars.size))
word = '阿道夫'
print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))
word = '阿'
print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))

id = 234
print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))
id = 234234
print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))

word = '阿道夫'
print('====> encoded result without split: {}'.format(vocab_chars.encode(word)))
print('====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
word = '阿 道 夫'
print('====> encoded result without split: {}'.format(vocab_chars.encode(word)))
print('====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
print('====> encoded result with split with reverse: {}'.format(vocab_chars.encode(word, split=True, reverse=True)))
word = '阿道 夫'
print('====> encoded result without split: {}'.format(vocab_chars.encode(word)))
print('====> encoded result with split: {}'.format(vocab_chars.encode(word, split=True)))
print('====> encoded result with split with reverse: {}'.format(vocab_chars.encode(word, split=True, reverse=True)))

'''
UE for UnicodeCharsVocabulary
'''
print('\n\n\tUE for UnicodeCharsVocabulary:')
vocab_file = '../data/vocab_seg_words_elmo.txt'
vocab_file1 = '../data/vocab_seg_chars_elmo.txt'
vocab_unicodechars = UnicodeCharsVocabulary(vocab_file, max_word_length=10, validate_file=True)
print('====> bos: {}'.format(vocab_chars.bos))
print('====> eos: {}'.format(vocab_chars.eos))
print('====> unk: {}'.format(vocab_chars.unk))
print('====> size: {}'.format(vocab_chars.size))

word = '阿道夫'
print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))
word = '阿'
print('====> word to id: {}'.format(vocab_chars.word_to_id(word)))

id = 234
print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))
id = 234234
print('====> id to word: {}'.format(vocab_chars.id_to_word(id)))

print('====> word chars ids: {}'.format(vocab_unicodechars.word_char_ids))
print('====> max word length: {}'.format(vocab_unicodechars.max_word_length))
words = '院子 中间 是 一颗 石榴树 ，'
print('====> word \t{}\t encoded result: {}'.format(words, vocab_chars.encode(words)))
print('====> word \t{}\t to char ids: {}'.format(words, vocab_unicodechars.word_to_char_ids(words)))
print('====> word \t{}\t encoded chars id result: {}'.format(words, vocab_unicodechars.encode_chars(words)))
ids = [1234, 3234, 22, 34, 341324, 21, 345]
print('====> decode \t{}\t to words: {}'.format(ids, vocab_unicodechars.decode(ids)))


'''
UE for Batcher
'''