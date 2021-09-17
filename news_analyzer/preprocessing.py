import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
# nltk.download('stopwords')
# nltk.download('punkt')


def preprocess(text: str) -> str:
    """
    Method for text preprocessing
    @param text: text to preprocess
    @return: preprocessed text
    """
    replace_by_space_re = re.compile("[\\/(){}\[\]\|@,;]")
    bad_symbols_re = re.compile(f"[^a-z\s]")
    stopwords_set = set(stopwords.words("english"))

    # Lowercase text
    text = text.lower()
    # Replace replace_by_space_re symbols by space in text
    text = re.sub(pattern=replace_by_space_re, repl=' ', string=text)
    # Delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub(pattern=bad_symbols_re, repl='', string=text)
    # Delete stopwords from text
    text_tokens = word_tokenize(text)
    text = ' '.join([word for word in text_tokens if word not in stopwords_set])

    return text


def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if preprocess(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


txt = 'Nvidia price target raised to $275 from $260 at BofA'
preprocess(text=txt)
