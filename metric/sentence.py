from RaTEScore import RaTEScore
'''
github: https://github.com/MAGIC-AI4Med/RaTEScore
paper : https://arxiv.org/pdf/2406.16845
似乎只有python==3.8才能解决包兼容的问题
暂时先不用
'''
pred_report = ['There are no intracranial hemorrhages.',
              'The musculature and soft tissues are intact.']

gt_report = ['There is no finding to suggest intracranial hemorrhage.',
            'The muscle compartments are intact.']

assert len(pred_report) == len(gt_report)

ratescore = RaTEScore()
# Add visualization_path here if you want to save the visualization result
# ratescore = RaTEScore(visualization_path = '')

scores = ratescore.compute_score(pred_report, gt_report)