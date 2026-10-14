import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from "react";
import type { AlgorithmSpec, Cell, Dataset, FrameReport, FrameSelection, Limits, ModelSummary, Sample } from "../api/types";

export type Stage = "source" | "frame" | "train" | "inspect" | "explain" | "probe";
export const STAGES: Stage[] = ["source", "frame", "train", "inspect", "explain", "probe"];

export interface Meta {
  limits: Limits;
  algorithms: AlgorithmSpec[];
  samples: Sample[];
}

export interface State {
  stage: Stage;
  dataset: Dataset | null;
  selection: FrameSelection | null;
  frameReport: FrameReport | null;
  runs: ModelSummary[];
  activeRunId: string | null;
  /** A row handed from the Inspect test-row browser to the Probe form. */
  probeSeed: { values: Record<string, Cell>; actual?: string; nonce: number } | null;
  /** The server no longer has our dataset/model (restart or eviction). */
  lost: boolean;
}

type Action =
  | { type: "dataset/loaded"; dataset: Dataset }
  | { type: "dataset/cleared" }
  | { type: "stage"; stage: Stage }
  | { type: "selection"; selection: FrameSelection }
  | { type: "frame/report"; report: FrameReport | null }
  | { type: "run/added"; run: ModelSummary }
  | { type: "run/removed"; id: string }
  | { type: "run/activate"; id: string }
  | { type: "probe/seed"; values: Record<string, Cell>; actual?: string }
  | { type: "lost" };

const initial: State = {
  stage: "source",
  dataset: null,
  selection: null,
  frameReport: null,
  runs: [],
  activeRunId: null,
  probeSeed: null,
  lost: false,
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "dataset/loaded":
      return { ...initial, dataset: action.dataset, stage: "source" };
    case "dataset/cleared":
      return { ...initial };
    case "stage":
      return { ...state, stage: action.stage };
    case "selection":
      return { ...state, selection: action.selection };
    case "frame/report":
      return { ...state, frameReport: action.report };
    case "run/added":
      return { ...state, runs: [...state.runs, action.run], activeRunId: action.run.id };
    case "run/removed": {
      const runs = state.runs.filter((r) => r.id !== action.id);
      const activeRunId = state.activeRunId === action.id ? (runs.at(-1)?.id ?? null) : state.activeRunId;
      return { ...state, runs, activeRunId };
    }
    case "run/activate":
      return { ...state, activeRunId: action.id };
    case "probe/seed":
      return {
        ...state,
        stage: "probe",
        probeSeed: { values: action.values, actual: action.actual, nonce: Date.now() },
      };
    case "lost":
      return { ...state, lost: true };
  }
}

const StateCtx = createContext<State>(initial);
const DispatchCtx = createContext<Dispatch<Action>>(() => {});
const MetaCtx = createContext<Meta | null>(null);

export function StoreProvider({ meta, children }: { meta: Meta; children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initial);
  return (
    <MetaCtx.Provider value={meta}>
      <StateCtx.Provider value={state}>
        <DispatchCtx.Provider value={dispatch}>{children}</DispatchCtx.Provider>
      </StateCtx.Provider>
    </MetaCtx.Provider>
  );
}

export const useStore = () => useContext(StateCtx);
export const useDispatch = () => useContext(DispatchCtx);
export function useMeta(): Meta {
  const meta = useContext(MetaCtx);
  if (!meta) throw new Error("useMeta outside provider");
  return meta;
}

export function useActiveRun(): ModelSummary | null {
  const { runs, activeRunId } = useStore();
  return runs.find((r) => r.id === activeRunId) ?? null;
}

/** Which stages can be opened, given what has been done so far. */
export function stageAvailability(state: State): Record<Stage, boolean> {
  const hasRun = state.activeRunId != null;
  return {
    source: true,
    frame: state.dataset != null,
    train: state.dataset != null && state.frameReport != null,
    inspect: hasRun,
    explain: hasRun,
    probe: hasRun,
  };
}
