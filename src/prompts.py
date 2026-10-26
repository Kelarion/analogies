## Define possible prompt building blocks

triplets = ("You will be presented with sequences of triplets. "
            "For example: a b c, d e f. ")
hint = "The last triplet will start with '{} {}' and you should provide the third word. "
nouns = "The tokens will be arbitrary nouns. "
structured = ("Your job is to complete the final query according to the structure. ")
only_final = "Output ONLY the next word. "



PROMPTS = {
    'sas': triplets+structured+only_final,
    'one_word': ("You will be presented with a sequence that has a certain pattern. "
                 "Your job is to provide the next word in the sequence, according to the "
                 "pattern. The pattern will be shared between sequences. Output ONLY the next word."),
}


DEFAULT_PROMPT = {'dialogue': 'sas', 'single': 'one_word', 'raw': None}


