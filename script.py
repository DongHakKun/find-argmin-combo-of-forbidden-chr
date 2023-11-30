import cupy as np


def one_hot_encoding(word):
    word_vec = np.zeros(26)

    for letter in list(set(word)):
        idx = (ord(letter)-ord('a'))%26
        word_vec[idx] = 1

    return word_vec


def one_hot_decoding(word_vec):
    decoded_word = []
    for i, val in enumerate(word_vec):
        if val == 1:
            decoded_word.append(chr(i + ord('a')))
    return decoded_word


def choose(t, k):
    if k == 0:
        return [[]]
    if not t:
        return []
    
    head = t[0]
    tail = t[1:]

    combos_with_head = [[head] + combo for combo in choose(tail, k-1)]
    combos_without_head = choose(tail, k)

    return combos_with_head + combos_without_head

#
with open('words.txt') as fin:
    words = [line.strip() for line in fin]
words_matrix = np.array([one_hot_encoding(word) for word in words])
del words # for memory save

#
combos = list(combinations('abcdefghijklmnopqrstuvwxyz', 5))
combos_matrix = np.array([one_hot_encoding(combo) for combo in combos]).T
del combos # for memory save

#
batch_size = 1000
max_iter_num = int(combos_matrix.shape[1]/batch_size)
best_combo = None
for i in range(max_iter_num):
    combos_matrix_batch = combos_matrix[:, :batch_size]

    if best_combo is not None:
        combos_matrix_batch = np.concatenate((best_combo, combos_matrix_batch), axis=1)

    count_matrix = words_matrix@combos_matrix_batch
    summed_vector = (count_matrix>0).sum(axis=0)
    argmin_idx = int(summed_vector.argmin())

    best_combo = combos_matrix_batch[:, [argmin_idx]]
    print(f'[{i}] The best combo is {one_hot_decoding(best_combo)} which does not exlude {1-summed_vector[argmin_idx]/words_matrix.shape[0]} in ratio')

    combos_matrix = combos_matrix[:, batch_size:] # for memory save
    np.get_default_memory_pool().free_all_blocks()
    np.get_default_pinned_memory_pool().free_all_blocks()

print('-'*100)
print(f'The best combo is {one_hot_decoding(best_combo)} which does not exlude {1-summed_vector[argmin_idx]/words_matrix.shape[0]} in ratio')
