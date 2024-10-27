import re
import nltk
import numpy as np
from nltk.corpus import wordnet as wn

class ObjectiveTest:

    def __init__(self, data, noOfQues):
        
        self.summary = data
        self.noOfQues = noOfQues

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        trivial_sentences = list()
        for sent in sentences:
            trivial = self.identify_trivial_sentences(sent)
            if trivial:
                trivial_sentences.append(trivial)
            else:
                continue
        return trivial_sentences

    def identify_trivial_sentences(self, sentence):
        # Ensure sentence is tokenized before tagging
        tokens = nltk.word_tokenize(sentence)
        pos_tokens = nltk.pos_tag(tokens)
        
        # Check if sentence length or first POS tag should exit early
        if pos_tokens[0][1] == "RB" or len(tokens) < 4:
            return None

        # Define the grammar for chunking noun phrases
        noun_phrases = []
        grammar = r"""
            CHUNK: {<NN>+<IN|DT>*<NN>+}
                {<NN>+<IN|DT>*<NNP>+}
                {<NNP>+<NNS>*}
        """
        chunker = nltk.RegexpParser(grammar)
        tree = chunker.parse(pos_tokens)

        # Extract noun phrases from the parsed tree
        for subtree in tree.subtrees():
            if subtree.label() == "CHUNK":
                temp = " ".join([sub[0] for sub in subtree])
                noun_phrases.append(temp.strip())

        # Identify words to replace in the sentence
        replace_nouns = []
        for word, _ in pos_tokens:
            for phrase in noun_phrases:
                if phrase[0] == '\'':
                    break
                if word in phrase:
                    replace_nouns.extend(phrase.split()[-2:])
                    break
            if not replace_nouns:
                replace_nouns.append(word)
            break

        # Return None if no replaceable nouns are found
        if not replace_nouns:
            return None

        # Find the shortest word length for Key
        val = min(len(i) for i in replace_nouns)

        # Prepare the trivial dictionary
        trivial = {
            "Answer": " ".join(replace_nouns),
            "Key": val
        }

        # Generate similar answer options if only one noun to replace
        if len(replace_nouns) == 1:
            trivial["Similar"] = self.answer_options(replace_nouns[0])
        else:
            trivial["Similar"] = []

        # Create the question with blanks
        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = "__________ " * len(replace_nouns)
        expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
        sentence = expression.sub(blanks_phrase.strip(), sentence, count=1)
        trivial["Question"] = sentence

        return trivial


    @staticmethod
    def answer_options(word):
        synsets = wn.synsets(word, pos="n")

        if len(synsets) == 0:
            return []
        else:
            synset = synsets[0]
        
        hypernym = synset.hypernyms()[0]
        hyponyms = hypernym.hyponyms()
        similar_words = []
        for hyponym in hyponyms:
            similar_word = hyponym.lemmas()[0].name().replace("_", " ")
            if similar_word != word:
                similar_words.append(similar_word)
            if len(similar_words) == 8:
                break
        return similar_words

    def generate_test(self):
        trivial_pair = self.get_trivial_sentences()
        question_answer = list()
        for que_ans_dict in trivial_pair:
            if que_ans_dict["Key"] > int(self.noOfQues):
                question_answer.append(que_ans_dict)
            else:
                continue
        question = list()
        answer = list()
        while len(question) < int(self.noOfQues):
            rand_num = np.random.randint(0, len(question_answer))
            if question_answer[rand_num]["Question"] not in question:
                question.append(question_answer[rand_num]["Question"])
                answer.append(question_answer[rand_num]["Answer"])
            else:
                continue
        return question, answer
