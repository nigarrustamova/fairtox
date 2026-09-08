"""Experiment 1a -- the classical floor.

The brief requires a clear baseline to beat or compare against. A fine-tuned
Transformer compared only against itself proves nothing about whether the
pretrained contextual representations were worth their cost, so the study is
anchored with TF-IDF plus logistic regression: minutes on CPU, no GPU, and a
real answer to "what does the deep model actually buy?"

It writes the same prediction-file format the neural runs do, so the fairness
auditor measures it with exactly the same code -- which lets the paper answer a
question it should: does the classical model show the same identity-term
disparity, or is this specifically a Transformer failure?
"""

from __future__ import annotations

from ..evaluation.metrics import classification_metrics, summarise
from ..evaluation.predictions import save_predictions
from ..registry import Tier, stage
from ..utils.logging_utils import get_logger
from ._common import get_split_fingerprint, get_splits

logger = get_logger(__name__)


@stage(
    "tfidf_baseline",
    tier=Tier.MUST,
    summary="Experiment 1a: TF-IDF and logistic regression floor (CPU only).",
    depends_on=("eda",),
)
def tfidf_baseline(ctx) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    config = ctx.config
    splits = get_splits(ctx)
    identity_columns = ctx.take("identity_columns") or []

    max_features = int(config.get("tfidf.max_features", 50000))
    ngram_max = int(config.get("tfidf.ngram_max", 2))
    min_df = int(config.get("tfidf.min_df", 2))

    # Fitted on train only. The vocabulary and the IDF weights are statistics,
    # and letting them see validation or test text is leakage.
    vectoriser = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vectoriser.fit_transform(splits["train"]["comment_text"].astype(str))
    logger.info(
        "TF-IDF fitted on train | %s rows x %s features",
        f"{x_train.shape[0]:,}", f"{x_train.shape[1]:,}",
    )

    classifier = LogisticRegression(
        max_iter=int(config.get("tfidf.max_iter", 1000)),
        C=float(config.get("tfidf.C", 1.0)),
        class_weight=config.get("tfidf.class_weight"),
        random_state=int(config.get("run.seed", 42)),
    )
    classifier.fit(x_train, splits["train"]["label"].to_numpy())

    threshold = float(config.get("evaluation.classification_threshold", 0.5))
    split_metrics = {}
    for split_name in ("val", "test"):
        part = splits[split_name]
        probs = classifier.predict_proba(
            vectoriser.transform(part["comment_text"].astype(str))
        )[:, 1]
        metrics = classification_metrics(part["label"].to_numpy(), probs, threshold)
        split_metrics[split_name] = metrics
        save_predictions(ctx.run_name, "tfidf", split_name, part, probs, identity_columns)
        logger.info("tfidf %s | %s", split_name, summarise(metrics))

    ctx.save_metrics(
        "tfidf_baseline",
        {
            "stage": "tfidf_baseline",
            "model_name": "tfidf",
            "model": "tfidf+logreg",
            "n_features": int(x_train.shape[1]),
            "split_fingerprint": get_split_fingerprint(ctx),
            "vectoriser": {
                "max_features": max_features,
                "ngram_range": [1, ngram_max],
                "min_df": min_df,
            },
            "metrics": split_metrics,
        },
    )
