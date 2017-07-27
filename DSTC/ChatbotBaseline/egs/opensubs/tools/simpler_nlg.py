NO_SPACE_BEFORE_WORDS = [".", ",", "'", "", "?"]
CAPITALIZE_WORDS = ["i"]
NO_SPACE_AFTER_APOSTROPHE = ["s", "t", "ll", "m", "re", "d"]

NO_SPACE_BEFORE_WORDS += ["'" + word for word in NO_SPACE_AFTER_APOSTROPHE]

class SimplerNLG:
    def __init__(self):
        pass

    @staticmethod
    def realise(words):
        ret = ""
        words = [word for word in words if word != "_UNK"]
        words.remove
        n_words = len(words)
        words.append("")
        for i in range(n_words):
            word = words[i]
            ret += SimplerNLG.realise_word(word)
            if SimplerNLG.should_add_space(words, i):
                ret += " "

        if ret != "":
            ret = ret[0].upper() + ret[1:]

        return ret

    @staticmethod
    def realise_word(word):
        if word in CAPITALIZE_WORDS:
            return word.upper()
        else:
            return word

    @staticmethod
    def should_add_space(words, index):
        w1 = words[index]
        w2 = words[index + 1]
        if w2 in NO_SPACE_BEFORE_WORDS:
            return False
        elif w1 == "'" and w2 in NO_SPACE_AFTER_APOSTROPHE:
            return False
        elif index != 0 and words[index - 1] == "o" and w1 == "'" and w2 == "clock":
            return False
        else:
            return True