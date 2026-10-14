// Plain-language definitions shown in <Term> tooltips. Written for someone
// who has never trained a model.

export const GLOSSARY = {
  accuracy: "Share of test rows the model labelled correctly. Can flatter a model when one class dominates.",
  balanced_accuracy:
    "The average of each class's hit rate (recall). A model that ignores a small class scores poorly here even if plain accuracy looks high.",
  precision: "Of the rows the model labelled as this class, how many really were. Low precision = many false alarms.",
  recall: "Of the rows that really are this class, how many the model found. Low recall = many misses.",
  f1: "A single score combining precision and recall (their harmonic mean). High only if both are high.",
  macro: "“Macro” averages treat every class equally, regardless of how many rows it has.",
  roc_auc:
    "How well the model ranks rows: the chance it scores a random positive row above a random negative one. 0.5 = coin flip, 1.0 = perfect ranking.",
  confusion_matrix:
    "A grid of real class (rows) vs predicted class (columns). The diagonal holds correct predictions; everything else is a specific kind of mistake.",
  cross_validation:
    "The training data is split into k folds; the model is trained k times, each time scored on a different held-out fold. The spread shows how stable the score is.",
  baseline:
    "A model that ignores the inputs and always predicts the most common class. Anything useful must beat it.",
  test_split:
    "Rows hidden from the model during training and only used to score it — a stand-in for new, unseen data.",
  stratified: "The split keeps each class's share the same in training and test sets.",
  permutation:
    "Shuffle one feature's values in the test set and measure how much the score drops. A big drop means the model leans on that feature.",
  association:
    "How much knowing this feature alone tells you about the target in the data (normalised mutual information). It describes the data, not the model.",
  impurity:
    "How much each feature's splits reduced class mixing inside the trees, on training data. Model-specific; biased toward features with many distinct values.",
  coefficients:
    "Size of each feature's weight in the linear model (on standardised inputs). Larger = stronger pull on the decision.",
  overfitting:
    "When a model memorises training rows instead of learning general patterns: great training score, worse test score.",
  balancing:
    "Weights rows so each class counts equally while training. Usually raises recall on small classes at some cost to overall accuracy.",
  extrapolation: "Predicting for values outside anything seen in training. The model is guessing beyond its experience.",
  imputation:
    "Filling a missing value. Here: the training median for numbers (plus a ‘was missing’ flag) and a ‘(missing)’ category for text.",
  sensitivity:
    "For this one prediction: how the probability changes if a feature is reset to a typical training value. Shows what the model reacted to — not what causes the outcome.",
  leakage:
    "A feature that contains the answer (e.g. recorded after the outcome). It makes scores look great but the model is useless in practice.",
} as const;

export type GlossaryKey = keyof typeof GLOSSARY;
