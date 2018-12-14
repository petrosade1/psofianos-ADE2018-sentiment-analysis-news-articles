from greek_stemmer import GreekStemmer
import sys
stemmer = GreekStemmer()
sys.stdout = open("With_neutral_stem.txt", "w")   
with open("With_neutral.txt") as f:               
    for line in f:
        for word in line.split(' '):
             word=word.strip()
             sys.stdout.write(stemmer.stem(word).lower()+' ')
             sys.stdout.flush()
        print('\n')

