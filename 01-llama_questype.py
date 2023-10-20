# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import pandas as pd
import fire
import random
from llama import Llama
from collections import Counter
import json
from tqdm import tqdm

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def find_first_word(words, string):
    first_word = None
    first_index = float('inf')
    for word in words:
        index = string.find(word)
        if index != -1 and index < first_index:
            first_word = word
            first_index = index
    return first_word

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    smart_data = json.load(open('dataset/VLAR-test.json','r'))['VLAR']


    id2type = {}
    types = ["path finding", "counting", "arithmetic", "algebra", "spatial reasoning", "measuring", "logic", "pattern"]
    for data in tqdm(smart_data[1:]):
        qid = data['Id']
        question = data['Question']
        text = f"There are eight question types: [counting, arithmetic, algebra, spatial reasoning, measuring, logic, path finding, ]。So, {question} which type does this question belong to?"

        
        dialogs = [[{"role": "user", "content": text}]]
        q_ts = []

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            # for msg in dialog:
                # print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            res = result['generation']['content']
            type_q = find_first_word(types, res)
            q_ts.append(type_q)
        max_type = most_common(q_ts)
        # print(
        #     f">question {id}: {max_type}"
        # )
        id2type[qid] = max_type
        # print("==================================")
    print("id2type file: save to ckpts/id2type.json")
    json.dump(id2type,open("ckpts/id2type.json",'w'))
    print("====================================================================================================")

if __name__ == "__main__":
    print("====================================================================================================")
    fire.Fire(main)

