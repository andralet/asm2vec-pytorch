# added by alex2000

import torch
import torch.nn as nn
import click
import asm2vec

import os
import re
import json

def cosine_similarity(v1, v2):
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()

def compare(ipath1, ipath2, mpath, epochs, device, lr):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model, tokens
    model, tokens = asm2vec.utils.load_model(mpath, device=device)
    functions, tokens_new = asm2vec.utils.load_data([ipath1, ipath2])
    tokens.update(tokens_new)
    model.update(2, tokens.size())
    model = model.to(device)
    
    # train function embedding
    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        epochs=epochs,
        device=device,
        mode='test',
        learning_rate=lr
    )

    # compare 2 function vectors
    v1, v2 = model.to('cpu').embeddings_f(torch.tensor([0, 1]))

    # print(f'cosine similarity : {cosine_similarity(v1, v2):.6f}')
    return cosine_similarity(v1, v2)

def extract_names(ipath):
    name = ipath
    file = ipath
    with open(ipath) as f:
        obj = re.match(r" \.name (?:[^.]*\.)?(?P<name>\S*)\n[^\n]*\n \.file (?P<file>\S*)", f.read())
        if obj is not None:
            res = obj.groupdict()
            if "name" in res:
                name = res["name"]
            if "file" in res:
                file = res["file"]
    return name, file

from tqdm import tqdm

@click.command()
@click.option('-i1', '--input1', 'idir1', help='target directory 1', required=True)
@click.option('-i2', '--input2', 'idir2', help='target directory 2', required=True)
@click.option('-m', '--model', 'mpath', help='model path', required=True)
@click.option('-o', '--output', 'output', default='resulting_pairs.json', help="json filepath for resulting pairs", show_default=True)
@click.option('-e', '--epochs', default=10, help='training epochs', show_default=True)
@click.option('-c', '--device', default='auto', help='hardware device to be used: cpu / cuda / auto', show_default=True)
@click.option('-lr', '--learning-rate', 'lr', default=0.02, help="learning rate", show_default=True)
def cli(idir1, idir2, mpath, output, epochs, device, lr):
    bad_results = []
    results_output = []
    for filename1 in os.listdir(idir1):
        f1 = os.path.join(idir1, filename1)
        if os.path.isfile(f1):
            name1, file1 = extract_names(f1)
            print(name1 + "@" + file1 + ":")
            results = []
            for filename2 in tqdm(os.listdir(idir2)):
                f2 = os.path.join(idir2, filename2)
                if os.path.isfile(f2):
                    name2, file2 = extract_names(f2)
                    if file1 != file2:
                        results.append((compare(f1, f2, mpath, epochs, device, lr), name2, file2))
                        # print("\t" + name2 + "@" + file2 + f': {compare(f1, f2, mpath, epochs, device, lr):.6f}')
            results.sort()
            results_output.append([[file1, name1], [results[-1][2], results[-1][1]]])
            print("\tTop - " + results[-1][1] + "@" + results[-1][2] + f': {results[-1][0]:.6f}')
            if len(results) > 1:
                print("\tSecond - " + results[-2][1] + "@" + results[-2][2] + f': {results[-2][0]:.6f}')
            if results[-1][1] != name1:
                bad_results.append(name1 + "@" + file1)
                for count, (res, name2, file2) in enumerate(results):
                    if name1 == name2:
                        print("\tExpected " + str(len(results) - count) + "(th) - " + name2 + "@" + file2 + f': {res:.6f}')
    with open(output, "w") as f:
        f.write(json.dumps(results_output))
    print("Failed for:")
    print(bad_results)

if __name__ == '__main__':
    cli()