import { useEffect, useState } from "react";
import TopNav from "./components/TopNav";
import SearchPortal from "./components/SearchPortal";
import { UI_COPY, resolveLang } from "./i18n";

const LANGUAGE_STORAGE_KEY = "sri-language";
const MODEL_BY_LANGUAGE = {
  en: "en_core_web_sm",
  es: "es_core_news_sm",
};

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

  useEffect(() => {
    const modelName = MODEL_BY_LANGUAGE[lang];

    fetch("http://127.0.0.1:5000/set_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelName }),
    })
      .then((response) => {
        if (!response.ok) {
          console.warn("Failed to set model on backend");
        }
      })
      .catch((error) => {
        console.error("Error setting model:", error);
      });
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
