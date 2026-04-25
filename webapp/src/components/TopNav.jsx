import { useEffect, useRef, useState } from "react";
import { GlobeIcon, GearIcon, FlagGB, FlagES } from "./icons";

function TopNav({ lang, onLanguageChange, copy }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  // Close menu when clicking outside
  useEffect(() => {
    function onDocClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("click", onDocClick);
    return () => document.removeEventListener("click", onDocClick);
  }, []);

  return (
    <header className="top-nav">
      <div className="top-nav-spacer" aria-hidden="true" />
      <h1 className="brand-wordmark">SRI</h1>

      <div className="top-nav-actions" role="group" aria-label={copy.topActionsAria}>
        <div className="lang-dropdown" ref={menuRef}>
          <button
            className="action-button lang-button is-active"
            onClick={() => setMenuOpen((v) => !v)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            <GlobeIcon />
            <span className="action-text">{copy.language}</span>
            <span className="caret">▾</span>
          </button>

          {menuOpen && (
            <ul className="lang-menu" role="menu">
              <li role="menuitem">
                <button className={`lang-option ${lang === "en" ? "selected" : ""}`} onClick={() => { onLanguageChange("en"); setMenuOpen(false); }}>
                  <FlagGB className="language-flag" /> {copy.englishName}
                </button>
              </li>
              <li role="menuitem">
                <button className={`lang-option ${lang === "es" ? "selected" : ""}`} onClick={() => { onLanguageChange("es"); setMenuOpen(false); }}>
                  <FlagES className="language-flag" /> {copy.spanishName}
                </button>
              </li>
            </ul>
          )}
        </div>

        <button className="action-button config-button" aria-label={copy.settings}>
          <GearIcon />
          <span className="action-text">{copy.settings}</span>
        </button>
      </div>
    </header>
  );
}

export default TopNav;
