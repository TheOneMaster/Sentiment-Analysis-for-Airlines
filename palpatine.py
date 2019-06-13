# Imports
import pandas as pd
from nltk import sent_tokenize
import numpy as np
from operator import itemgetter
import pickle
import re

class Sentiment():
    
    def __init__(self):
        # The pickle file has the keywords for all words in all given languages
        with open('keywords.pickle', 'rb') as keywords:
            self.lang_keywords = pickle.load(keywords)
            self.W_INCR = 0.293
            self.W_DECR = -0.293
            
    @staticmethod
    def clean_sentence(text):
        """
        Removes parts of a text irrelevant to sentiment analysis (Ex: links, numbers, @)
        """
        text = re.sub(r"(?:\@|https?\://)\S+", "", text)
        text = re.sub(r'[0-9\.]+', '', text)
        text = text.replace('"', '')
        text = text.replace("''", '')
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        return text

    def polarity(self, text, language):
        """Returns a dictionary containing the polarity of the text. Shows the different proprtions
        of sentiments expressed in the text.
        
        Attributes:
        
        tweet - The text that is to be translated. A string that can consist of multiple sentences
        
        language - The language of the text to be translated. If not in the present in the model, will
                   return the string "N/A"
        """
        
        if language not in self.lang_keywords:
            return 'N/A'
        
        keywords = self.lang_keywords[language]
        
        sentences = sent_tokenize(text)
        sentences_clean = [self.clean_sentence(x) for x in sentences]
        sentences_clean = [x for x in sentences_clean if x not in ('', ' ')]
        
        if len(sentences_clean) == 0:
            return 0
        
        pol_scores = []
        
        for sentence in sentences_clean:
            words = sentence.split()
            total = len(words)
            upper = 0
            
            pos = 0
            neg = 0
            neu = 0
            
            for word in words:
                test = word.isupper()
                if test:
                    upper += 1
                    
                word = word.lower()
                if word in keywords:
                    if keywords[word] == self.W_INCR:
                        if test:
                            pos += 2
                        else:
                            pos += 1
                    else:
                        if test:
                            neg += 2
                        else:
                            neg += 1
                else:
                    neu += 1
            
            total_pos = pos/total
            total_neu = neu/total
            total_neg = neg/total
            pol_scores.append((total_pos, total_neu, total_neg, upper, total))
            
        total_pos = round(sum(list(map(itemgetter(0), pol_scores)))/len(sentences_clean), 3)
        total_neu = round(sum(list(map(itemgetter(1), pol_scores)))/len(sentences_clean), 3)
        total_neg = round(sum(list(map(itemgetter(2), pol_scores)))/len(sentences_clean), 3)
        upper = sum(list(map(itemgetter(3), pol_scores)))
        total = sum(list(map(itemgetter(4), pol_scores)))
        
        
        compound = ((total_pos - total_neg)*2)*(1 + upper/total)
        if compound > 1:
            compound = 1
        elif compound < -1:
            compound = -1
        
        pol_dict = {
            'pos': total_pos,
            'neu': total_neu,
            'neg': total_neg,
            'compound': compound
        }
        
        return pol_dict
                       
if __name__ == '__main__':
    print('This is a demo of the module. It is used for languages other than english. It is currently unifinished.')
    test = 'Vous etes mon petite amie, je vous aime.'
    sent = Sentiment()
    polarity = sent.polarity(text=test, language='fr')
    print(polarity)
