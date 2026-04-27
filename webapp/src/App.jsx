import { useEffect, useState } from "react";
import TopNav from "./components/TopNav";
import SearchPortal from "./components/SearchPortal";
import { UI_COPY, resolveLang } from "./i18n";

const LANGUAGE_STORAGE_KEY = "sri-language";

function App() {
  const [lang, setLang] = useState(() => {
    if (typeof window === "undefined") {
      return "es";
    }

    return resolveLang(window.localStorage.getItem(LANGUAGE_STORAGE_KEY));
  });
  const copy = UI_COPY[lang];

  useEffect(() => {
    document.documentElement.lang = lang;

    try {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, lang);
    } catch (error) {
      console.warn("Unable to persist selected language", error);
    }
  }, [lang]);

  return (
    <div className="app-root">
      <div className="page-card" role="application" aria-label={copy.appAriaLabel}>
        <TopNav lang={lang} onLanguageChange={setLang} copy={copy} />

        {/* Contenido central de la pantalla de prueba. */}
        <main className="page-content">
          <SearchPortal copy={copy} />
        </main>
      </div>
    </div>
  );
}

export default App;
