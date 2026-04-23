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
  const [activeFilters, setActiveFilters] = useState(new Set());
  const [results, setResults] = useState([]);
  const [totalResults, setTotalResults] = useState(0);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const pageSize = 5;

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
    if (!query || query.trim().length === 0) return;
    doSearch(query.trim());
  };

  const doSearch = async (q) => {
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:5000/search?query=${encodeURIComponent(q)}&top_k=80`);
      const data = await res.json();
      setResults(data.results || []);
      setTotalResults(data.total || 0);
      setPage(1);
    } catch (err) {
      console.error("Error buscando:", err);
    } finally {
      setLoading(false);
    }
  };

  const toggleFilter = (label) => {
    setActiveFilters((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
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
          <FilterPill
            key={filter.label}
            icon={filter.icon}
            label={filter.label}
            active={activeFilters.has(filter.label)}
            onToggle={() => toggleFilter(filter.label)}
          />
        ))}
      </div>

      {loading ? (
        <p className="results-hint">Buscando "{query}" ...</p>
      ) : results && results.length > 0 ? (
        <>
          <p className="results-count">Se encontraron {totalResults} resultados para "{query}"</p>

          <div className="results-list">
            {results.slice((page - 1) * pageSize, page * pageSize).map((item) => (
              <article className="result-item" key={item.doc_id}>
                <div className="result-left">
                  <div className="result-icon">
                    {item.file_type === "PDF" ? <PdfIcon /> : item.file_type === "IMAGEN" ? <ImageIcon /> : item.file_type === "VIDEO" ? <VideoIcon /> : item.file_type === "DOCUMENTO" ? <FolderIcon /> : <FileIcon />}
                  </div>
                </div>

                <div className="result-body">
                  <a href={item.url} target="_blank" rel="noreferrer" className="result-title">{item.title}</a>
                  <p className="result-snippet">{item.snippet}</p>
                  <div className="result-meta">{item.domain} • {item.crawl_date ? item.crawl_date.split("T")[0] : ""}</div>
                </div>

                <div className="result-right">
                  <span className="result-tag">{item.file_type}</span>
                </div>
              </article>
            ))}
          </div>

          <div className="results-pager">
            <button onClick={() => setPage(Math.max(1, page - 1))} disabled={page === 1}>{'<'}</button>
            <span className="pager-info">{page}</span>
            <button onClick={() => setPage(page + 1)} disabled={page * pageSize >= results.length}>{'>'}</button>
          </div>
        </>
      ) : (
        <p className="results-hint">Los resultados aparecerán aquí</p>
      )}
    </section>
  );
}

export default SearchPortal;
