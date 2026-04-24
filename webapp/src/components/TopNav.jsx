import { useEffect, useRef, useState } from "react";
import { GlobeIcon, GearIcon, FlagGB, FlagES } from "./icons";

function TopNav() {
  const [lang, setLang] = useState("en"); // default English
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

  // Call backend to set spaCy model when language changes
  useEffect(() => {
    const modelName = lang === "en" ? "en_core_web_sm" : "es_core_news_sm";
    try {
      fetch("http://127.0.0.1:5000/set_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelName }),
      }).then((r) => {
        if (!r.ok) console.warn("Failed to set model on backend");
      }).catch((err) => console.error("Error setting model:", err));
    } catch (err) {
      console.error("Error sending set_model request:", err);
    }
  }, [lang]);

  return (
    <header className="top-nav">
      <div className="top-nav-spacer" aria-hidden="true" />
      <h1 className="brand-wordmark">SRI</h1>

      <div className="top-nav-actions" role="group" aria-label="Top actions">
        <div className="lang-dropdown" ref={menuRef}>
          <button
            className={`action-button lang-button ${lang === "en" ? "is-active" : ""}`}
            onClick={() => setMenuOpen((v) => !v)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            <GlobeIcon />
            <span className="action-text">Idioma</span>
            <span className="caret">▾</span>
          </button>

          {menuOpen && (
            <ul className="lang-menu" role="menu">
              <li role="menuitem">
                <button className={`lang-option ${lang === "en" ? "selected" : ""}`} onClick={() => { setLang("en"); setMenuOpen(false); }}>
                  <FlagGB className="language-flag" /> Inglés
                </button>
              </li>
              <li role="menuitem">
                <button className={`lang-option ${lang === "es" ? "selected" : ""}`} onClick={() => { setLang("es"); setMenuOpen(false); }}>
                  <FlagES className="language-flag" /> Español
                </button>
              </li>
            </ul>
          )}
        </div>

        <button className="action-button config-button" aria-label="Configuración">
          <GearIcon />
          <span className="action-text">Configuración</span>
        </button>
      </div>
    </header>
  );
}

export default TopNav;
