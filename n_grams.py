import re
import sys


def most_common_n_grams(poems, n, number_of_n_grams):
    n_gram_counts = {}

    for poem_index in range(len(poems)):
        poem = poems[poem_index]
        for word_index in range(len(poem)):
            if word_index >= n:
                n_gram_list = poem[word_index - n:word_index]
                n_gram = ""

                for word in n_gram_list:
                    n_gram += word + " "

                try:
                    n_gram_counts[n_gram] += 1
                except KeyError:
                    n_gram_counts[n_gram] = 1

    # print("Part 1 done")

    keys = []
    for key in n_gram_counts.keys():
        keys.append(key)

    for n_gram in keys:
        if n_gram_counts[n_gram] <= 3:
            del n_gram_counts[n_gram]

    print("Number of total n grams: %d" % len(n_gram_counts))
    # print("Number of times 'of life' appears: %d" % n_gram_counts["of life "])

    while len(n_gram_counts) > number_of_n_grams:
        min_count = sys.maxsize
        min_n_gram = None
        for n_gram, count in n_gram_counts.items():
            if count < min_count:
                min_n_gram = n_gram
                min_count = n_gram_counts[min_n_gram]

        del n_gram_counts[min_n_gram]

    print("N gram counts: " + str(n_gram_counts))

    n_grams = []
    for n_gram in n_gram_counts.keys():
        n_grams.append(n_gram)

    return n_grams


def get_n_gram_proportions(poems, n_grams, n):
    proportions = []

    for i in range(len(poems)):
        proportions.append({})
        for n_gram in n_grams:
            proportions[i][n_gram] = 0

    for poem_index in range(len(poems)):
        poem = poems[poem_index]
        for word_index in range(len(poem)):
            n_gram_list = poem[word_index - n:word_index]

            n_gram = ""

            for word in n_gram_list:
                n_gram += word + " "

            try:
                proportions[poem_index][n_gram] += 1
            except KeyError:
                pass

        for n_gram in proportions[poem_index].keys():
            proportions[poem_index][n_gram] /= len(poem)

    return proportions


def get_poem_label_pairs(poems):
    poem_list = poems.split("\n")
    poems_labels_list = []
    row = 0
    for poem in poem_list:
        if row != 0:
            poem_label_pair = poem.split(",")
            poems_labels_list.append(poem_label_pair)
        row += 1
    # print(poems_labels_list)
    return poems_labels_list


def main():
    n_gram_count = int(sys.argv[1])  # how many n grams are frequencies calculated for
    n = int(sys.argv[2])  # number of words in an n gram

    file = open("poems_reordered_further_cleaned.csv", 'r')
    if file.mode == 'r':
        contents = file.read()
    poem_label_pair_list = get_poem_label_pairs(contents)

    labels = []
    poem_texts = []
    for poem in poem_label_pair_list:
        poem_texts.append(poem[1])
        labels.append(poem[0])

    poem_words = []

    for poem in poem_texts:
        word_list = re.split("\s+", poem)
        for word in word_list:
            if len(word) == 0:
                word_list.remove(word)
            if word == 'ff':
                word_list.remove(word)
            if len(word) == 1 and word != 'o' and word != 'i' and word != 'a':
                word_list.remove(word)

        poem_words.append(word_list)

    n_grams = most_common_n_grams(poem_words, n, n_gram_count)
    n_gram_frequencies = get_n_gram_proportions(poem_words, n_grams, n)
    # print("n gram frequencies for poem 1: " + str(n_gram_frequencies[0]))

    filename = 'poem_ngram_attributes.csv'
    file = open(filename, 'w')

    file.write("Century,")

    for i in range(n_gram_count):
        file.write("%d-gram %d" % (n, i))
        if i != n_gram_count - 1:
            file.write(",")

    file.write("\n")

    for i in range(len(poem_words)):
        file.write(labels[i] + ",")
        f_count = 0
        frequencies = n_gram_frequencies[i]
        for n_gram in frequencies.keys():
            file.write("%.3f" % frequencies[n_gram])
            if f_count != n_gram_count - 1:
                file.write(",")
            f_count += 1

        if i != len(poem_words) - 1:
            file.write("\n")


main()
