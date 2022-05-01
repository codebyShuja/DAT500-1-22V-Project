import re
from datetime import datetime
import spacy
from mrjob.job import MRJob
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from contractions import CONTRACTION_MAP

nlp = spacy.load("en_core_web_sm")
stopword_list = stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


class MRPROCESSING(MRJob):

    def RemoveSpecialCharacters(self, text, removeNumber=True):
        pattern = r'[^a-zA-z0-9\s]' if not removeNumber else r'[^a-zA-z\s]'
        output = re.sub(pattern, '', text)
        return output

    def TokenizerIs(self, outputT):
        tokens = []
        TokensOutput = outputT.split()
        TokensOutput = [token.strip() for token in TokensOutput]
        return TokensOutput

    def ContractionsAre(self, text, ContMapping=MAPP):
        contractions_pattern = re.compile('({})'.format('|'.join(ContMapping.keys())), flags=re.IGNORECASE | re.DOTALL)

        def MatchesAre(Cont):
            match = Cont.group(0)
            first_char = match[0]
            ExCont = ContMapping.get(match) \
                if ContMapping.get(match) \
                else ContMapping.get(match.lower())
            ExCont = first_char + ExCont[1:]
            return ExCont

        exText = ConPattern.sub(expand_match, text)
        exText = re.sub("'", "", expanded_text)
        return expanded_text

    def RMDuplicateChar(self, tokens):
        Rpattern = re.compile(r'(\w*)(\w)\2(\w*)')
        sameSub = r'\1\2\3'

        def replace(oldS):
            if wn.synsets(oldS):
                return oldS
            Nword = repeat_pattern.sub(match_substitution, oldS)
            return replace(oldS) if oldS != oldS else oldS

        CorToken = [replace(word) for word in tokens]
        return CorToken

    def RMStopWords(self, tokensARE):
        FtOKEN = [tokensARE for token in tokensARE if token not in stopword_list]
        return FtOKEN

    def lemmatizeString(self, ST):
        ST = nlp(ST)
        ST = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.ST
                         for word in ST])
        return ST

    def mapperFunction(self, _, line):
        line = self.ContractionsAre(line)
        line = self.RemoveSpecialCharacters(line)
        line = line.lower()
        line = self.lemmatizeString(line)
        tokensaree = self.TokenizerIs(line)
        tokensaree = self.RMDuplicateChar(tokensaree)
        tokensaree = self.RMStopWords(tokensaree)
        processed_rev = ' '.join(tokensaree)

        yield (processed_rev), None


if __name__ == '__main__':
    start_time = datetime.now()
    MR_Preprocess.run()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Total Executing Time : ",elapsed_time)
