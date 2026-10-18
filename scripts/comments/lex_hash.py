import re

from files import Lang
from lex_cxx import LineIndex, Span

_NEXT = {Lang.CCL: re.compile(r"""[#"'/]"""), Lang.PAR: re.compile(r'[#"]')}
_QUOTED = {
    '"': re.compile(r'"(?:[^"\\]|\\.)*"?', re.S),
    "'": re.compile(r"'(?:[^'\\]|\\.)*'?", re.S),
}


def lex(text: str, lang: Lang) -> list[Span]:
    pattern = _NEXT[lang]
    index = LineIndex(text)
    spans = []
    i = 0
    while (m := pattern.search(text, i)) is not None:
        i = m.start()
        c = text[i]
        if c == "#":
            newline = text.find("\n", i)
            end = len(text) if newline < 0 else newline
            if end > i and text[end - 1] == "\r":
                end -= 1
            spans.append(Span("hash", i, end, index.line_of(i), text[i:end]))
            i = end
        elif c == "/":
            if text.startswith("/*", i):
                close = text.find("*/", i + 2)
                end = len(text) if close < 0 else close + 2
                spans.append(Span("block", i, end, index.line_of(i), text[i:end]))
                i = end
            else:
                i += 1
        else:
            i = _QUOTED[c].match(text, i).end()
    return spans
