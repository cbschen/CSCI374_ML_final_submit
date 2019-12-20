import nltk
import csv
import re
import pandas
from textblob import TextBlob
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


def write_atts_to_csv(labels, sentiments, unique_word_ratios, poem_lengths, avg_word_lens, parts_of_speech):
    filename = 'poem_attributes.csv'
    file = open(filename, 'w')

    file.write("Century,Sentiments,Word Diversity,Number of Words,Average Word Length"
               ",Number of Adjectives,Number of Adpositions,"
               "Number of Adverbs,Number of Conjunctions,Number of Determiners,Number of Nouns,Number of Numerals,"
               "Number of Particles,Number of Pronouns,Number of Verbs"
               "\n")

    for i in range(len(labels)):
        file.write(str(labels[i]) + ',' + str(sentiments[i]) + ',' + str(unique_word_ratios[i]) + ',' +
                   str(poem_lengths[i]) + ',' + str(avg_word_lens[i]) +
                   ',' + str(parts_of_speech[i]['ADJ']) + ',' + str(parts_of_speech[i]['ADP'])
                   + ',' + str(parts_of_speech[i]['ADV']) + ',' + str(parts_of_speech[i]['CONJ']) + ',' +
                   str(parts_of_speech[i]['DET']) + ',' + str(parts_of_speech[i]['NOUN']) + ',' +
                   str(parts_of_speech[i]['NUM']) + ',' + str(parts_of_speech[i]['PRT']) + ',' +
                   str(parts_of_speech[i]['PRON']) + ',' + str(parts_of_speech[i]['VERB']) + '\n')


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


# Returns a list of all poems in the format of a list of stanzas
# Pass in the list of poems
def split_stanzas(poems):
    poem_stanzas = []
    for poem in poems:
        poem_stanzas.append(re.split('[^\s]\s{3}[^\s]', poem))
    return poem_stanzas


# Returns list of average stanza length for each poem
def avg_stanza_len(poem_stanzas):
    avg_stnz_lens = []
    for poem in poem_stanzas:
        total_words = 0
        for stanza in poem:
            total_words += len(stanza)
        avg_stnz_lens.append(total_words / len(poem))
    return avg_stnz_lens


# Returns ratio of unique words to total words in poem
def get_word_diversity(poems):
    # Make a running list of each unique word
    word_diversities = []
    for poem in poems:
        unique_words = []
        for word in poem:
            if word not in unique_words:
                unique_words.append(word)
        word_diversities.append(len(unique_words) / len(poem))
    return word_diversities


# returns the counts of each part of speech in the poems, as determined by the nltk pos tagger
def pos_counts(poems):
    pos_proportion = []
    c = 0
    for poem in poems:
        try:
            tagged_poem = nltk.pos_tag(poem, tagset='universal')
        except IndexError:
            print("Problem: " + str(poem))

        if c == 359:
            print("Tagged poem: " + str(tagged_poem))

        pos_count = {'ADJ': 0, 'ADP': 0, 'ADV': 0, 'CONJ': 0, 'DET': 0,
                     'NOUN': 0, 'NUM': 0, 'PRT': 0, 'PRON': 0, 'VERB': 0, 'X': 0, '.': 0}

        for word in tagged_poem:
            nltk_tag = word[1]
            # pos = condense_pos(nltk_tag)
            pos_count[nltk_tag] += 1

        for pos in pos_count:
            pos_count[pos] /= len(poem)

        pos_proportion.append(pos_count)

        c += 1

    return pos_proportion


def get_pos_tags(poems):
    pos_tags = []

    for poem in poems:
        try:
            tagged_poem = nltk.pos_tag(poem, tagset='universal')
        except IndexError:
            print("Problem: " + str(poem))

        for tagged_word in tagged_poem:
            pos_tags.append(tagged_word[1])

    return pos_tags


def get_poem_lens(poems):
    poem_lens = []
    for poem in poems:
        poem_lens.append(len(poem))
    return poem_lens


def get_avg_word_lens(poems):
    # poem_words = []
    avg_word_lens = []
    for poem in poems:
        total_letters = 0
        for word in poem:
            total_letters += len(word)
        avg_word_lens.append(total_letters / len(poem))
    return avg_word_lens


def analyze_sentiment(poems):
    sentiments = []
    for poem in poems:
        poem_blob = TextBlob(poem)
        sentiments.append(poem_blob.sentiment.polarity)

    return sentiments


def main():
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

    # poem_stanzas = split_stanzas(poem_texts)

    # for poem1 in poem_stanzas:
    #     for stanza in poem1:
    #         print("stanza: " + stanza)
    #     print("\nNEW POEM:")

    # ----------CODE FOR NON-N-GRAM ATTRIBUTES-------------

    # sentiments = analyze_sentiment(poem_texts)
    # print("Sentiments: " + str(sentiments))
    # unique_word_ratios = get_word_diversity(poem_words)
    # print("Word Diversity: " + str(unique_word_ratios))
    # poem_lengths = get_poem_lens(poem_words)
    # print("Poem word counts: " + str(poem_lengths))
    # avg_word_lens = get_avg_word_lens(poem_words)
    # print("Average word lengths: " + str(avg_word_lens))
    # # print("Pos counts: " + str(pos_counts(poem_texts)))
    # parts_of_speech = pos_counts(poem_words)
    # # print("Parts of speech: " + str(parts_of_speech[359]))
    # # pos_tags = get_pos_tags(poem_words)

    # -----CODE FOR N-GRAM ATTRIBUTES--------
    n_gram_count = 200
    n_grams = most_common_n_grams(poem_words, 3, n_gram_count)
    # print("n grams: " + str(n_grams))
    n_gram_frequencies = get_n_gram_proportions(poem_words, n_grams, 3)
    print("n gram frequencies for poem 1: " + str(n_gram_frequencies[0]))

    # Write non-n-gram attributes to csv
    # write_atts_to_csv(labels, sentiments, unique_word_ratios, poem_lengths, avg_word_lens, parts_of_speech)

    filename = 'poem_attributes.csv'
    file = open(filename, 'w')

    file.write("Century,")

    for i in range(n_gram_count):
        file.write("2-gram %d" % i)
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
