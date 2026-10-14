import { useId, useLayoutEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { Issue, Kind } from "../api/types";
import { GLOSSARY, type GlossaryKey } from "../lib/glossary";

/* ---------- Buttons ---------- */

type Variant = "primary" | "accent" | "ghost" | "quiet";
export function Button({
  variant = "ghost",
  size = "md",
  busy = false,
  className = "",
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md" | "lg"; busy?: boolean }) {
  return (
    <button
      type="button"
      className={`btn btn-${variant} btn-${size} ${busy ? "is-busy" : ""} ${className}`}
      aria-busy={busy || undefined}
      {...rest}
      disabled={rest.disabled || busy}
    >
      {busy && <Pulse small />}
      {children}
    </button>
  );
}

/* ---------- Floating tooltip (portal, viewport-aware) ---------- */

export function Tip({ content, children, className = "" }: { content: ReactNode; children: ReactNode; className?: string }) {
  const id = useId();
  const anchor = useRef<HTMLSpanElement>(null);
  const bubble = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; below: boolean } | null>(null);

  useLayoutEffect(() => {
    if (!open || !anchor.current || !bubble.current) return;
    const a = anchor.current.getBoundingClientRect();
    const b = bubble.current.getBoundingClientRect();
    const below = a.top - b.height - 10 < 8;
    const left = Math.min(Math.max(8, a.left + a.width / 2 - b.width / 2), window.innerWidth - b.width - 8);
    setPos({ left, top: below ? a.bottom + 8 : a.top - b.height - 8, below });
  }, [open]);

  return (
    <span
      ref={anchor}
      className={`tip-anchor ${className}`}
      tabIndex={0}
      aria-describedby={open ? id : undefined}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => {
        setOpen(false);
        setPos(null);
      }}
      onFocus={() => setOpen(true)}
      onBlur={() => {
        setOpen(false);
        setPos(null);
      }}
    >
      {children}
      {open &&
        createPortal(
          <div
            ref={bubble}
            id={id}
            role="tooltip"
            className={`tip ${pos?.below ? "tip-below" : ""}`}
            style={pos ? { left: pos.left, top: pos.top } : { left: -9999, top: 0 }}
          >
            {content}
          </div>,
          document.body,
        )}
    </span>
  );
}

/** An inline technical term with a plain-language definition on hover/focus. */
export function Term({ k, children }: { k: GlossaryKey; children: ReactNode }) {
  return (
    <Tip content={GLOSSARY[k]} className="term">
      {children}
    </Tip>
  );
}

export function InfoDot({ children }: { children: ReactNode }) {
  return (
    <Tip content={children} className="info-dot">
      <span aria-label="More information">?</span>
    </Tip>
  );
}

/* ---------- Signal pulse (loading indicator) ---------- */

export function Pulse({ small = false, label }: { small?: boolean; label?: string }) {
  return (
    <span className={`pulse ${small ? "pulse-sm" : ""}`} role="status" aria-label={label ?? "Loading"}>
      <svg viewBox="0 0 60 20" aria-hidden="true">
        <path d="M0 10 H14 L18 3 L24 17 L29 6 L33 12 L36 10 H60" />
      </svg>
    </span>
  );
}

/* ---------- Issues / notices ---------- */

const GLYPH: Record<string, ReactNode> = {
  info: <circle cx="8" cy="8" r="6.2" />,
  warn: <path d="M8 1.8 15 14.2H1z" />,
  error: <path d="M3 3l10 10M13 3 3 13" />,
  good: <path d="M2.5 8.5 6.3 12 13.5 4" />,
};

export function Glyph({ level }: { level: string }) {
  return (
    <svg className={`glyph glyph-${level}`} viewBox="0 0 16 16" aria-hidden="true">
      {GLYPH[level] ?? GLYPH.info}
    </svg>
  );
}

export function Notice({ level, title, children, action }: { level: Issue["level"] | "good"; title?: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className={`notice notice-${level}`} role={level === "error" ? "alert" : undefined}>
      <Glyph level={level} />
      <div className="notice-body">
        {title && <strong className="notice-title">{title}</strong>}
        {children && <div className="notice-text">{children}</div>}
      </div>
      {action && <div className="notice-action">{action}</div>}
    </div>
  );
}

export function IssueList({ issues, actions }: { issues: Issue[]; actions?: (issue: Issue) => ReactNode }) {
  if (!issues.length) return null;
  const order = { error: 0, warn: 1, info: 2 } as const;
  const sorted = [...issues].sort((a, b) => order[a.level] - order[b.level]);
  return (
    <ul className="issue-list">
      {sorted.map((issue, i) => (
        <li key={`${issue.code}-${issue.column ?? i}`} className={`issue issue-${issue.level}`}>
          <Glyph level={issue.level} />
          <span className="issue-text">{issue.message}</span>
          {actions?.(issue)}
        </li>
      ))}
    </ul>
  );
}

/* ---------- Form controls ---------- */

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: ReactNode; disabled?: boolean }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          disabled={o.disabled}
          className={o.value === value ? "is-on" : ""}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, children, disabled }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode; disabled?: boolean }) {
  return (
    <label className={`toggle ${disabled ? "is-disabled" : ""}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-track" aria-hidden="true">
        <span className="toggle-thumb" />
      </span>
      <span className="toggle-label">{children}</span>
    </label>
  );
}

/* ---------- Labels & headings ---------- */

const KIND_LABEL: Record<Kind, string> = { numeric: "NUM", categorical: "CAT", identifier: "ID", text: "TEXT", datetime: "DATE" };
const KIND_HELP: Record<Kind, string> = {
  numeric: "Numeric — measured quantities. Gaps are filled with the training median.",
  categorical: "Categorical — a fixed set of labels, one-hot encoded for the model.",
  identifier: "Identifier — (almost) unique per row, like an ID. Not useful for learning patterns.",
  text: "Free text — long, mostly unique strings. Only usable as a coarse category here.",
  datetime: "Date/time — not used directly; derive month, weekday etc. as separate columns.",
};

export function KindBadge({ kind, as }: { kind: Kind; as?: "categorical" }) {
  const shown = as ?? kind;
  return (
    <Tip content={KIND_HELP[shown]} className={`kind kind-${shown}`}>
      {KIND_LABEL[shown]}
    </Tip>
  );
}

export function SectionHead({ index, title, kicker, aside }: { index: string; title: ReactNode; kicker?: ReactNode; aside?: ReactNode }) {
  return (
    <header className="section-head">
      <div>
        <span className="section-index">§ {index}</span>
        <h2 className="section-title">{title}</h2>
        {kicker && <p className="section-kicker">{kicker}</p>}
      </div>
      {aside && <div className="section-aside">{aside}</div>}
    </header>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <svg viewBox="0 0 120 40" aria-hidden="true" className="empty-trace">
        <path d="M0 20 H120" />
      </svg>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

/** Thin labelled meter used inline in tables (0..1). */
export function Meter({ value, tone = "ink", label }: { value: number; tone?: "ink" | "signal" | "teal" | "amber"; label?: string }) {
  const v = Math.max(0, Math.min(1, value));
  return (
    <span className={`meter meter-${tone}`} role="meter" aria-valuenow={v} aria-valuemin={0} aria-valuemax={1} aria-label={label}>
      <span style={{ width: `${v * 100}%` }} />
    </span>
  );
}
