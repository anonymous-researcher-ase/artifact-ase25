import argparse
import os
import json
import openai
from time import sleep
import random
from typing import List, Dict
from tqdm import tqdm

from numpy import dot
from numpy.linalg import norm

openai.api_type = "azure"
openai.azure_endpoint = "https://hzfsls.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = ""  # azure openai api key


def get_embedding(lang, text):
    # text = text.replace("\n", " ")
    return (
        openai.embeddings.create(input=[text], model="text-embedding-ada-002")
        .data[0]
        .embedding
    )


def get_cosine_sim(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


with open("./Data/snippet/Var.json", "r", encoding="utf-8") as f:
    snippets = json.load(f)

embeddings = {}

for snippet in tqdm(snippets):
    embeddings[snippet] = {
        "code": snippets[snippet],
        "embedding": get_embedding(snippet.split(" ")[0], snippets[snippet]),
    }

with open("./Result/openai_embeddings_Var.json", "w", encoding="utf-8") as f:
    json.dump(embeddings, f, indent=4)

# to tsv

with open(f"./TSV_Result/openai_embeddings_Var.tsv", "w", encoding="utf-8") as f:
    for name in embeddings:
        f.write("\t".join([str(x) for x in embeddings[name]["embedding"]]) + "\n")

# tsv metadata

with open(
    f"./TSV_Result/openai_embeddings_metadata_Var.tsv", "w", encoding="utf-8"
) as f:
    f.write("Name" + "\t" + "Language" + "\t" + "Feature" + "\n")
    for name in embeddings:
        f.write(
            name.replace(" ", "_")
            + "\t"
            + name.split(" ")[0]
            + "\t"
            + name.split(" ")[1]
            + "\n"
        )
