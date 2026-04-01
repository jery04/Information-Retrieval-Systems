export function NetworkLogoIcon() {
  return (
    <svg viewBox="0 0 96 96" aria-hidden="true" className="network-logo">
      {/* Nodos y conexiones del isotipo central. */}
      <g fill="none" stroke="currentColor" strokeWidth="5.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M28 24L50 16L70 28L68 52L47 64L27 53L26 34Z" />
        <path d="M27 34L47 46L68 52" />
        <path d="M50 16L47 46" />
        <path d="M70 28L47 46" />
      </g>
      <g fill="currentColor">
        <circle cx="28" cy="24" r="6" />
        <circle cx="50" cy="16" r="6" />
        <circle cx="70" cy="28" r="6" />
        <circle cx="68" cy="52" r="6" />
        <circle cx="47" cy="64" r="6" />
        <circle cx="27" cy="53" r="6" />
        <circle cx="26" cy="34" r="6" />
        <circle cx="47" cy="46" r="6" />
      </g>
    </svg>
  );
}

export function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon icon-search">
      <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M20 20L16.25 16.25" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function FolderIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon">
      <path
        d="M3 7.5A2.5 2.5 0 015.5 5h4l2 2H18.5A2.5 2.5 0 0121 9.5v7A2.5 2.5 0 0118.5 19h-13A2.5 2.5 0 013 16.5v-9z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function ImageIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon">
      <rect x="3" y="4" width="18" height="16" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <circle cx="9" cy="9" r="1.6" fill="currentColor" />
      <path d="M6 16L11 11L14.5 14.5L17 12L20 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function VideoIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon">
      <rect x="3" y="4" width="18" height="16" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.8" />
      <path d="M10 9L15 12L10 15V9z" fill="currentColor" />
    </svg>
  );
}

export function PdfIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon">
      <path d="M7 3h7l5 5v12.5A1.5 1.5 0 0117.5 22h-10A1.5 1.5 0 016 20.5V4.5A1.5 1.5 0 017.5 3z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M14 3v5h5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M8.25 16h1.5a1.4 1.4 0 100-2.8h-1.5V18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12.5 18v-4.8h1.3a2.4 2.4 0 010 4.8h-1.3z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M17 13.2h-2.2V18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function FileIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon">
      <path d="M7 3h7l5 5v12.5A1.5 1.5 0 0117.5 22h-10A1.5 1.5 0 016 20.5V4.5A1.5 1.5 0 017.5 3z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M14 3v5h5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M9 13h6M9 16h6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
