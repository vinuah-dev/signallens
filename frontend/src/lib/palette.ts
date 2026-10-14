// Class colours, chosen to stay distinguishable on the paper background and
// for common colour-vision deficiencies in the first four slots.
const CLASS_COLOURS = ["#1D6E73", "#E0531F", "#6B4E8A", "#B98A1C", "#3F6FB5", "#7A8B2E", "#C2477B", "#5C5C55"];

export function classColour(labels: string[], label: string): string {
  const i = Math.max(0, labels.indexOf(label));
  return CLASS_COLOURS[i % CLASS_COLOURS.length];
}

// Distinct hues for comparing model runs (separate from class colours).
const RUN_COLOURS = ["#191A17", "#E0531F", "#1D6E73", "#6B4E8A", "#B98A1C", "#3F6FB5"];
export const runColour = (index: number) => RUN_COLOURS[index % RUN_COLOURS.length];
