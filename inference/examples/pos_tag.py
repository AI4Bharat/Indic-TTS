import nltk
nltk.download('averaged_perceptron_tagger')

INPUT_SENTENCE = "Hello my name is Gokul and I am from Madras"
print("Input:", INPUT_SENTENCE)
pos_tags = nltk.tag.pos_tag(INPUT_SENTENCE.split())
print(pos_tags)
