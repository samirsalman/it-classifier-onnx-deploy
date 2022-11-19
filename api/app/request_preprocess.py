import re
from app.api_exeptions import EmptyText


def preprocess_text(text: str):
    if not text:
        raise EmptyText()

    if not isinstance(text, str):
        text = str(text)

    # lowercase
    text = text.lower()
    # remove paragraph numbers
    text = re.sub("\[\d*\]", "", text.strip())
    # removing new lines
    text = text.replace("\n", " ")
    # removing tabs
    text = text.replace("\t", " ")
    # removing unicode chars
    text = re.sub("[^\w\s]", "", text)
    text = text.lstrip()
    return text
