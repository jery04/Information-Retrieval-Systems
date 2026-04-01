import { useEffect, useState } from "react";
import FilterPill from "./FilterPill";
import {
  FileIcon,
  FolderIcon,
  ImageIcon,
  NetworkLogoIcon,
  PdfIcon,
  SearchIcon,
  VideoIcon,
} from "./icons";

const FILTERS = [
  { label: "Documentos", icon: <FolderIcon /> },
  { label: "Imagenes", icon: <ImageIcon /> },
  { label: "Videos", icon: <VideoIcon /> },
  { label: "PDFs", icon: <PdfIcon /> },
  { label: "Otros", icon: <FileIcon /> },
];

const SUBTITLE_OPTIONS = [
  "Tu algoritmo con complejo de Sherlock :)",
  "Tu módulo oficial de cazar bits :)",
  "Tu excavadora de datos enterrados :)",
  "Tu herramienta de búsqueda :)",
  "Tu radar semántico de alta precisión :)",
  "Tu indexador con espíritu de arqueólogo digital :)",
  "Tu asistente de investigación :)",
];

function SearchPortal() {
  const [query, setQuery] = useState("");
  const [subtitleIndex, setSubtitleIndex] = useState(0);
  const [isFading, setIsFading] = useState(false);

  useEffect(() => {
    let changeTimeoutId;

    const intervalId = setInterval(() => {
      // Desvanece y luego cambia el texto para transicionar suavemente.
      setIsFading(true);

      changeTimeoutId = setTimeout(() => {
        setSubtitleIndex((currentIndex) => (currentIndex + 1) % SUBTITLE_OPTIONS.length);
        setIsFading(false);
      }, 280);
    }, 3000);

    return () => {
      clearInterval(intervalId);

      if (changeTimeoutId) {
        clearTimeout(changeTimeoutId);
      }
    };
  }, []);

  const handleSubmit = (event) => {
    // Este prototipo no envia datos, solo conserva la maqueta visual.
    event.preventDefault();
  };

  return (
    <section className="search-portal" aria-label="Portal de busqueda">
      <NetworkLogoIcon />

      <h2 className="portal-title">BÚSQUEDA AVANZADA</h2>
  <p className={`portal-subtitle ${isFading ? "is-fading" : ""}`}>{SUBTITLE_OPTIONS[subtitleIndex]}</p>

      <form className="search-form" onSubmit={handleSubmit}>
        <SearchIcon />

        <input
          type="text"
          className="search-input"
          placeholder="Buscar informacion..."
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-label="Caja de busqueda"
        />

        <button type="submit" className="search-button">
          BUSCAR
        </button>
      </form>

      <div className="filters-row" aria-label="Filtros por tipo">
        {FILTERS.map((filter) => (
          <FilterPill key={filter.label} icon={filter.icon} label={filter.label} />
        ))}
      </div>

      <p className="results-hint">Los resultados aparecerán aquí</p>
    </section>
  );
}

export default SearchPortal;
