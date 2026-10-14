/** Landing illustration: rows of a table pass through the lens and come out
 * as a prediction with its reasons — the whole product in one picture.
 * Purely decorative (aria-hidden); motion is disabled for reduced-motion users. */
export function HeroLens() {
  const rows = [0, 1, 2, 3, 4, 5];
  const rowY = (i: number) => 74 + i * 26;
  return (
    <svg className="hero-lens" viewBox="0 0 460 300" aria-hidden="true">
      {/* the table */}
      <g className="hl-table">
        <rect x="8" y="44" width="128" height="18" className="hl-head" />
        {rows.map((i) => (
          <g key={i} className="hl-row" style={{ animationDelay: `${i * 90}ms` }}>
            <rect x="8" y={rowY(i) - 8} width="40" height="16" />
            <rect x="52" y={rowY(i) - 8} width="40" height="16" className={i === 2 ? "hl-missing" : ""} />
            <rect x="96" y={rowY(i) - 8} width="40" height="16" />
            <rect x="12" y={rowY(i) - 2} width={10 + ((i * 7) % 22)} height="4" className="hl-val" />
            {i !== 2 && <rect x="56" y={rowY(i) - 2} width={8 + ((i * 11) % 26)} height="4" className="hl-val" />}
            <rect x="100" y={rowY(i) - 2} width={6 + ((i * 5) % 28)} height="4" className="hl-val hl-val-alt" />
          </g>
        ))}
        <text x="8" y="34" className="hl-caption">
          your CSV
        </text>
      </g>

      {/* rays converging into the lens and fanning out to the answer */}
      {rows.map((i) => (
        <path key={`in${i}`} className="hl-ray" d={`M140 ${rowY(i)} C 180 ${rowY(i)}, 190 150, 212 150`} />
      ))}
      <path className="hl-ray hl-ray-out" d="M290 150 C 310 150, 316 110, 330 104" />
      <path className="hl-ray hl-ray-out" d="M290 150 C 310 150, 316 196, 330 202" />

      {/* the lens */}
      <g className="hl-lens">
        <circle cx="250" cy="150" r="46" className="hl-ring" />
        <circle cx="250" cy="150" r="36" className="hl-inner" />
        <path className="hl-trace" d="M206 150 h14 l6 -14 l9 28 l8 -20 l7 10 l5 -4 h33" />
        <text x="250" y="216" textAnchor="middle" className="hl-caption">
          trained model
        </text>
      </g>

      {/* the answer */}
      <g className="hl-answer">
        <text x="336" y="70" className="hl-caption">
          prediction
        </text>
        <text x="336" y="98" className="hl-label">
          Yes
        </text>
        <rect x="336" y="104" width="110" height="9" className="hl-track" />
        <rect x="336" y="104" width="90" height="9" className="hl-bar hl-bar-a" />
        <text x="446" y="98" textAnchor="end" className="hl-num">
          82%
        </text>
        <text x="336" y="132" className="hl-label">
          No
        </text>
        <rect x="336" y="138" width="110" height="9" className="hl-track" />
        <rect x="336" y="138" width="20" height="9" className="hl-bar hl-bar-b" />
        <text x="446" y="132" textAnchor="end" className="hl-num">
          18%
        </text>

        <text x="336" y="186" className="hl-caption">
          because
        </text>
        <line x1="391" x2="391" y1="194" y2="246" className="hl-zero" />
        <text x="385" y="207" textAnchor="end" className="hl-why">
          contract
        </text>
        <rect x="391" y="199" width="46" height="9" className="hl-push hl-push-1" />
        <text x="385" y="225" textAnchor="end" className="hl-why">
          tenure
        </text>
        <rect x="391" y="217" width="28" height="9" className="hl-push hl-push-2" />
        <text x="385" y="243" textAnchor="end" className="hl-why">
          tickets
        </text>
        <rect x="391" y="235" width="12" height="9" className="hl-push hl-push-3" />
      </g>
    </svg>
  );
}
