"""Token-overlap sentence support, shared by the output guard, the judge and the mock review team."""
import re

_STOP = frozenset(
    "a an the is are was were be been at of for to in on and or with against by as it its from that this than then if "
    "can may must will would should not no do does did has have had which who whom what when where how".split()
)


def content_tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def sentence_support(sentence, evidence_tokens):
    toks = content_tokens(sentence)
    return 1.0 if not toks else len(toks & evidence_tokens) / len(toks)
