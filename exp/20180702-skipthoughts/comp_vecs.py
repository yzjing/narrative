import skipthoughts
import os
import pickle

# model = skipthoughts.load_model()
# encoder = skipthoughts.Encoder(model)

# book_path = '../data/Gutenberg'
# book_list = [item for item in os.listdir(book_path) if '.txt' in item]

# for book in book_list[0:5]:
# 	sentences = [line.decode('utf8').strip() for line in open(os.path.join(book_path, book)).readlines()]
# 	sentences = [item for item in sentences if len(item) > 0]
# 	vecs = encoder.encode(sentences)
# 	pickle.dump(vecs, open(book + '_vecs.p', 'wb'))

v = pickle.load(open('Hamlin Garland___The Spirit of Sweetwater.txt_vecs.p', 'rb'))
print(v)