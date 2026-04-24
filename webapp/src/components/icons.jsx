export function NetworkLogoIcon() {
  return (
    <img src="/logo.png" alt="SRI logo" className="network-logo" />
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

export function GlobeIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon" width="20" height="20">
      <path d="M12 2a10 10 0 100 20 10 10 0 000-20z" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M2 12h20M12 2c2.5 3 2.5 9 0 20M12 2c-2.5 3-2.5 9 0 20" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="icon" width="20" height="20">
      <path d="M3 6h14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="20" cy="6" r="1.6" fill="currentColor" />

      <path d="M3 12h10" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="16" cy="12" r="1.6" fill="currentColor" />

      <path d="M3 18h8" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="14" cy="18" r="1.6" fill="currentColor" />
    </svg>
  );
}

export function FlagGB({ className }) {
  return (
    <svg
      className={className || "language-flag"}
      width="20"
      height="14"
      viewBox="0 0 60 36"
      aria-hidden="true"
      role="img"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="60" height="36" fill="#012169" />
      <path d="M0 14h60v8H0z" fill="#fff" />
      <path d="M0 15h60v6H0z" fill="#cf142b" />
      <path d="M26 0h8v36h-8z" fill="#fff" />
      <path d="M28 0h4v36h-4z" fill="#cf142b" />
      <path d="M0 0L60 36M60 0L0 36" stroke="#fff" strokeWidth="6" />
      <path d="M0 0L60 36M60 0L0 36" stroke="#cf142b" strokeWidth="3" />
    </svg>
  );
}

export function FlagES({ className }) {
  return (
    <svg
      className={className || "language-flag"}
      width="20"
      height="14"
      viewBox="0 0 3 2"
      aria-hidden="true"
      role="img"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="3" height="2" fill="#C60B1E" />
      <rect y="0.4" width="3" height="1.2" fill="#FFC400" />
      <rect width="3" height="0.4" fill="#C60B1E" />
    </svg>
  );
}
