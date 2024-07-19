from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import time
def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    gts = {1: ['a cat on a mat'], 2: ['a dog is running'], 3: ['a bird is flying'], 4: ['a fish swims in the water']}
    res = {1: ['a cat on a mat'], 2: ['a dog is running'], 3: ['a bird is flying'], 4: ['a fish swims in the water']}
    输出
    {'BLEU_1': 1.0, 'BLEU_2': 1.0, 'BLEU_3': 1.0, 'BLEU_4': 1.0, 'METEOR': 1.0, 'ROUGE_L': 1.0, 'CIDEr': 1.0}
    """
    # post-processing, make format consistent
    for k in res.keys():
        res[k][0] = (res[k][0]+' ').replace('. ', ' . ').replace(' - ', '-')

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L"),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

if __name__ == '__main__':
    # 开始计时

    gts = {1: ['a cat on a mat'], 2: ['a dog is running'], 3: ['a bird is flying'], 4: ['a fish swims in the water']}
    res = {1: ['a cat on a mat'], 2: ['a dog is running'], 3: ['a bird is flying'], 4: ['a fish swims in the water']}
    start = time.time()
    print(compute_scores(gts, res))
    print('Time elapsed: %.2fs' % (time.time() - start))
    # {'BLEU_1': 1.0, 'BLEU_2': 1.0, 'BLEU_3': 1.0, 'BLEU_4': 1.0, 'METEOR': 1.0, 'ROUGE_L': 1.0, 'CIDEr': 1.0}