import random

def generate_qa_data(n=100):
    contexts = [
        "John went to the market to buy vegetables. He saw his friend Tom there.",
        "Alice was reading a mystery novel in the library when she heard a loud noise.",
        "Tommy loves playing football every evening at the school ground.",
        "Sara baked a chocolate cake for her brother's birthday.",
        "David and Emma went to the zoo last weekend to see the lions."
    ]

    qa_pairs = []
    for i in range(n):
        context = random.choice(contexts)
        if "market" in context:
            q = "Where did John go?"
            a = "the market"
            start = context.index(a)
        elif "library" in context:
            q = "Where was Alice reading?"
            a = "in the library"
            start = context.index(a)
        elif "football" in context:
            q = "What does Tommy love to play?"
            a = "football"
            start = context.index(a)
        elif "cake" in context:
            q = "What did Sara bake?"
            a = "a chocolate cake"
            start = context.index(a)
        else:
            q = "Where did David and Emma go?"
            a = "to the zoo"
            start = context.index(a)

        qa = {
            "title": "Story_" + str(i),
            "paragraphs": [
                {
                    "context": context,
                    "qas": [
                        {
                            "id": f"q{i}",
                            "question": q,
                            "answers": [{"text": a, "answer_start": start}],
                            "is_impossible": False
                        }
                    ]
                }
            ]
        }
        qa_pairs.append(qa)
    return {"data": qa_pairs}

import json
with open("generated_qa.json", "w") as f:
    json.dump(generate_qa_data(200), f, indent=2)
