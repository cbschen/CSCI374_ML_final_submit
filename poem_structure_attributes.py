import nltk
import re
from textblob import TextBlob


def write_atts_to_csv(labels, sentiments, unique_word_ratios, poem_lengths, avg_word_lens, parts_of_speech):
    filename = 'poem_structure_attributes.csv'
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

    # ----------CODE FOR NON-N-GRAM ATTRIBUTES-------------

    sentiments = analyze_sentiment(poem_texts)
    print("Sentiments: " + str(sentiments))

    unique_word_ratios = get_word_diversity(poem_words)
    print("Word Diversity: " + str(unique_word_ratios))

    poem_lengths = get_poem_lens(poem_words)
    print("Poem word counts: " + str(poem_lengths))

    avg_word_lens = get_avg_word_lens(poem_words)
    print("Average word lengths: " + str(avg_word_lens))

    # print("Pos counts: " + str(pos_counts(poem_texts)))
    parts_of_speech = pos_counts(poem_words)

    # Write attributes to csv
    write_atts_to_csv(labels, sentiments, unique_word_ratios, poem_lengths, avg_word_lens, parts_of_speech)


main()
