// Mirrors the JSON produced by backend/signallens. Kept hand-written and
// small rather than generated, so it doubles as documentation of the API.

export type Cell = string | number | boolean | null;
export type Kind = "numeric" | "categorical" | "identifier" | "text" | "datetime";
export type FeatureKind = "numeric" | "categorical";
export type Level = "info" | "warn" | "error";

export interface Issue {
  level: Level;
  code: string;
  message: string;
  column?: string | null;
}

export interface HistogramBins {
  type: "bins";
  bins: { x0: number; x1: number; count: number }[];
}
export interface HistogramBars {
  type: "bars";
  bars: { label: string; count: number }[];
  other?: number;
}

export interface ColumnProfile {
  name: string;
  kind: Kind;
  dtype: string;
  missing: number;
  missing_pct: number;
  unique: number;
  unique_pct: number;
  is_constant: boolean;
  high_cardinality: boolean;
  low_cardinality_numeric: boolean;
  target_candidate: boolean;
  stats: Record<string, number | string | boolean | null>;
  histogram: HistogramBins | HistogramBars;
  sample_values: Cell[];
}

export interface Profile {
  rows: number;
  columns: number;
  missing_cells: number;
  missing_pct: number;
  rows_with_missing: number;
  duplicate_rows: number;
  kind_counts: Record<Kind, number>;
  column_profiles: ColumnProfile[];
  suggested_target: string | null;
}

export interface RowsPage {
  total: number;
  offset: number;
  limit: number;
  columns: string[];
  rows: { index: number; values: Cell[] }[];
}

export interface Dataset {
  id: string;
  filename: string;
  source: "upload" | "sample";
  meta: { delimiter: string; encoding: string; bytes: number };
  warnings: Issue[];
  profile: Profile;
  preview: RowsPage;
}

export interface Sample {
  key: string;
  title: string;
  blurb: string;
  tags: string[];
  bytes: number;
}

export interface Limits {
  max_upload_bytes: number;
  max_rows: number;
  max_columns: number;
  min_rows: number;
  max_classes: number;
  extensions: string[];
  test_size: { min: number; max: number; default: number };
}

export interface FrameSelection {
  target: string;
  features: string[];
  categorical_overrides: string[];
}

export interface ClassCount {
  label: string;
  count: number;
}

export interface FrameReport {
  target: string;
  features: string[];
  kinds: Record<string, FeatureKind>;
  rows_used: number;
  rows_dropped: number;
  classes: ClassCount[];
  imbalance_ratio: number;
  imbalanced: boolean;
  association: { feature: string; score: number }[];
  issues: Issue[];
}

export interface ParamSpec {
  name: string;
  label: string;
  type: "int" | "float";
  default: number | null;
  min: number;
  max: number;
  help: string;
  log: boolean;
  nullable: boolean;
  step: number | null;
}

export interface AlgorithmSpec {
  key: string;
  label: string;
  summary: string;
  strengths: string;
  native_importance: "impurity" | "coefficients" | null;
  params: ParamSpec[];
}

export interface TrainSettings {
  algorithm: string;
  params: Record<string, number | null>;
  test_size: number;
  balance: boolean;
  random_state: number;
}

export interface Metrics {
  accuracy: number;
  balanced_accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  precision_weighted: number;
  recall_weighted: number;
  f1_weighted: number;
  roc_auc?: number;
}

export interface PerClass {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface Evaluation {
  metrics: Metrics;
  train_metrics: { accuracy: number; balanced_accuracy: number; f1_macro: number };
  baseline: { strategy: string; label: string; accuracy: number; balanced_accuracy: number; f1_macro: number };
  per_class: PerClass[];
  confusion: { labels: string[]; matrix: number[][] };
  roc: { positive_label: string; auc: number; points: { fpr: number; tpr: number }[] } | null;
  n_train: number;
  n_test: number;
}

export interface NumericFeature {
  name: string;
  kind: "numeric";
  min: number;
  max: number;
  p01: number;
  p99: number;
  median: number;
  mean: number;
  std: number;
  integer: boolean;
  missing_rate: number;
}

export interface CategoricalFeature {
  name: string;
  kind: "categorical";
  categories: { value: string; count: number }[];
  n_categories: number;
  mode: string | null;
  missing_rate: number;
}

export type FeatureSpec = NumericFeature | CategoricalFeature;

export interface TestExample {
  row: number;
  values: Record<string, Cell>;
  actual: string;
  predicted: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface CrossValidation {
  metric?: string;
  folds?: number;
  scores: number[];
  mean?: number;
  std?: number;
  skipped?: string;
}

export interface ModelSummary {
  id: string;
  dataset_id: string;
  created_at: number;
  algorithm: string;
  algorithm_label: string;
  params: Record<string, number | null>;
  balance: boolean;
  test_size: number;
  random_state: number;
  split_key: string;
  target: string;
  features: string[];
  kinds: Record<string, FeatureKind>;
  labels: string[];
  frame: { rows_used: number; rows_dropped: number; classes: ClassCount[]; imbalanced: boolean; imbalance_ratio: number };
  evaluation: Evaluation;
  cross_validation: CrossValidation | null;
  interpretation: { tone: "good" | "warn" | "info"; text: string }[];
  importance: {
    permutation: { metric: string; rows: number; repeats: number; items: { feature: string; mean: number; std: number }[] };
    native: { kind: "impurity" | "coefficients"; items: { feature: string; share: number }[] } | null;
  };
  association: { feature: string; score: number }[];
  schema: FeatureSpec[];
  test_examples: { truncated: boolean; total: number; rows: TestExample[] };
  warnings: Issue[];
  timing: { fit_seconds: number; total_seconds: number };
}

export interface PredictionNote {
  level: "info" | "warn";
  feature: string;
  code: string;
  message: string;
}

export interface Sensitivity {
  feature: string;
  value: Cell;
  typical: Cell;
  probability_if_typical: number;
  delta: number;
}

export interface Prediction {
  predicted: string;
  confidence: number;
  probabilities: { label: string; probability: number }[];
  inputs: Record<string, Cell>;
  notes: PredictionNote[];
  sensitivity?: Sensitivity[];
}

export interface WhatIf {
  feature: string;
  kind: FeatureKind;
  x: (number | string)[];
  series: { label: string; values: number[] }[];
  current: number | string | null;
}
