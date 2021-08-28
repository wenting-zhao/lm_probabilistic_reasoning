"""R^2 metric."""

from sklearn.metrics import r2_score, mean_absolute_error

import datasets

class Regress(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="regression metrics",
            citation="",
            homepage="https://github.com/mjpost/sacreBLEU",
            inputs_description="",
            features=datasets.Features({
                'predictions': datasets.Value('float32'),
                'references': datasets.Value('float32'),
            }),
            codebase_urls=["https://github.com/mjpost/sacreBLEU"],
            reference_urls=["https://github.com/mjpost/sacreBLEU",
                            "https://en.wikipedia.org/wiki/BLEU",
                            "https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213"]
        )

    def _compute(self, predictions, references, sample_weight=None, multioutput="uniform_average"):
        return {
            "r2": r2_score(
                references,
                predictions,
                sample_weight=sample_weight,
                multioutput=multioutput
            ).item(),
            "mae": mean_absolute_error(
                references,
                predictions,
                sample_weight=sample_weight,
                multioutput=multioutput
            ).item(),
        }
