import json
from .ce import CheXbertMetrics
from .nlg import compute_scores
# import .sentence import RaTEScore

class Metric():
    '''
    input: gts:[scentence], res:[scentence]
    return: ce_scores, nlg_scores
    '''
    def __init__(self, checkpoint_path='./checkpoints/checkpoint.pth', mbatch_size=16, device='cuda'):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.ceMetric = CheXbertMetrics(self.checkpoint_path,self.mbatch_size,self.device)
    def compute(self, gts, res):
        ce_scores = self.ceMetric.compute(gts, res)
        gts = {i: [gt] for i, gt in enumerate(gts)}
        res = {i: [re] for i, re in enumerate(res)}
        nlg_scores = compute_scores(gts, res)
        scores = {}
        scores.update(ce_scores)
        scores.update(nlg_scores)
        return scores, ce_scores, nlg_scores
    def __del__(self):
        pass

# if __name__ == '__main__':
#     checkpoint_path = '/root/MRG/checkpoints/checkpoint.pth'
#     metric = Metric(checkpoint_path, 16, 'cuda')
#     gts = ['There are no intracranial hemorrhages.',
#             'The musculature and soft tissues are intact.']
#     res = ['There is no finding to suggest intracranial hemorrhage.',
#             'The muscle compartments are intact.']
    
#     _,ce_scores, nlg_scores = metric.compute(gts, res)
#     print(ce_scores)
#     print(nlg_scores)