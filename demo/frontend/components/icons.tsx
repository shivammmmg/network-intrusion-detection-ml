interface IconProps {
  className?: string;
}

const base = {
  width: 18,
  height: 18,
  viewBox: "0 0 20 20",
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const
};

export function FlowIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <circle cx="4" cy="10" r="2.1" />
      <circle cx="10" cy="4.5" r="2.1" />
      <circle cx="10" cy="15.5" r="2.1" />
      <circle cx="16" cy="10" r="2.1" />
      <line x1="6" y1="9" x2="8" y2="6" />
      <line x1="6" y1="11" x2="8" y2="14" />
      <line x1="12" y1="5" x2="14" y2="8.5" />
      <line x1="12" y1="14.5" x2="14" y2="11.5" />
    </svg>
  );
}

export function BarsIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <line x1="4" y1="16" x2="4" y2="11" />
      <line x1="10" y1="16" x2="10" y2="6" />
      <line x1="16" y1="16" x2="16" y2="3" />
      <line x1="2" y1="17.5" x2="18" y2="17.5" strokeWidth={1.2} opacity={0.5} />
    </svg>
  );
}

export function SearchIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <circle cx="8.5" cy="8.5" r="5.5" />
      <line x1="16.5" y1="16.5" x2="12.6" y2="12.6" />
    </svg>
  );
}

export function EyeIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <path d="M2 10c2-3.5 5.5-5.5 8-5.5s6 2 8 5.5c-2 3.5-5.5 5.5-8 5.5s-6-2-8-5.5Z" />
      <circle cx="10" cy="10" r="2.2" />
    </svg>
  );
}

export function WaveIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <path d="M2 12c1.5-4 3-6 5-6s3.5 8 5.5 8 3-6 5.5-6" />
    </svg>
  );
}

export function GaugeIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <path d="M3 14a7 7 0 0 1 14 0" />
      <line x1="10" y1="14" x2="13.2" y2="9.5" />
      <circle cx="10" cy="14" r="1" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function AlertIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <path d="M10 3.5 17.5 16h-15L10 3.5Z" />
      <line x1="10" y1="8.5" x2="10" y2="12" />
      <circle cx="10" cy="14.3" r="0.8" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function DiceIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <rect x="3" y="3" width="14" height="14" rx="2.5" />
      <circle cx="7" cy="7" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="13" cy="7" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="10" cy="10" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="7" cy="13" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="13" cy="13" r="0.9" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function CheckIcon({ className }: IconProps) {
  return (
    <svg className={className} {...base} aria-hidden="true">
      <circle cx="10" cy="10" r="7.5" />
      <polyline points="6.5 10.2 8.8 12.5 13.5 7.5" />
    </svg>
  );
}
