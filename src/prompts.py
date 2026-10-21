PROMPTS = {
    'sas': ("You will be presented with sequences of triplets: a, b, c. "
            "They will follow a certain structure, and your job is to complete the final "
            "triplet according to that structure. Output ONLY the your answer."),
    'one_word': ("You will be presented with a sequence that has a certain pattern. "
                 "Your job is to provide the next word in the sequence, according to the "
                 "pattern. The pattern will be shared between sequences. Output ONLY the next word."),
}
DEFAULT_PROMPT = {'dialogue': 'sas', 'single': 'one_word', 'raw': None}


