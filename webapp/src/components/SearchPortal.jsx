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
  { id: "documents", icon: <FolderIcon /> },
  { id: "images", icon: <ImageIcon /> },
  { id: "videos", icon: <VideoIcon /> },
  { id: "pdfs", icon: <PdfIcon /> },
  { id: "other", icon: <FileIcon /> },
];

const FILE_TYPE_TO_KEY = {
  PDF: "pdf",
  IMAGEN: "image",
  IMAGE: "image",
  VIDEO: "video",
  DOCUMENTO: "document",
  DOCUMENT: "document",
  OTRO: "other",
  OTHER: "other",
};

function getFileTypeKey(rawType) {
  const normalizedType = String(rawType || "").toUpperCase();
  return FILE_TYPE_TO_KEY[normalizedType] || "other";
}

function getFileTypeIcon(rawType) {
  const typeKey = getFileTypeKey(rawType);

  if (typeKey === "pdf") {
    return <PdfIcon />;
  }

  if (typeKey === "image") {
    return <ImageIcon />;
  }

  if (typeKey === "video") {
    return <VideoIcon />;
  }

  if (typeKey === "document") {
    return <FolderIcon />;
  }

  return <FileIcon />;
}

function getLocalizedFileType(rawType, copy) {
  const typeKey = getFileTypeKey(rawType);

  return copy.fileTypes[typeKey] || rawType || copy.fileTypes.other;
}

function SearchPortal({ copy }) {
  const [query, setQuery] = useState("");
  const [subtitleIndex, setSubtitleIndex] = useState(0);
  const [isFading, setIsFading] = useState(false);
  const [activeFilters, setActiveFilters] = useState(new Set());
  const [results, setResults] = useState([]);
  const [totalResults, setTotalResults] = useState(0);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const pageSize = 6;

  useEffect(() => {
    setSubtitleIndex((currentIndex) => currentIndex % copy.subtitles.length);
  }, [copy.subtitles.length]);

  useEffect(() => {
    let changeTimeoutId;

    const intervalId = setInterval(() => {
      // Desvanece y luego cambia el texto para transicionar suavemente.
      setIsFading(true);

      changeTimeoutId = setTimeout(() => {
        setSubtitleIndex((currentIndex) => (currentIndex + 1) % copy.subtitles.length);
        setIsFading(false);
      }, 280);
    }, 3000);

    return () => {
      clearInterval(intervalId);

      if (changeTimeoutId) {
        clearTimeout(changeTimeoutId);
      }
    };
  }, [copy.subtitles.length]);

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

  const toggleFilter = (filterId) => {
    setActiveFilters((prev) => {
      const next = new Set(prev);
      if (next.has(filterId)) {
        next.delete(filterId);
      } else {
        next.add(filterId);
      }
      return next;
    });
  };

  return (
    <section className="search-portal" aria-label={copy.searchPortalAria}>
      <NetworkLogoIcon />

      <h2 className="portal-title">{copy.title}</h2>
      <p className={`portal-subtitle ${isFading ? "is-fading" : ""}`}>{copy.subtitles[subtitleIndex]}</p>

      <form className="search-form" onSubmit={handleSubmit}>
        <SearchIcon />

        <input
          type="text"
          className="search-input"
          placeholder={copy.searchPlaceholder}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-label={copy.searchInputAria}
        />

        <button type="submit" className="search-button">
          {copy.searchButton}
        </button>
      </form>

      <div className="filters-row" aria-label={copy.filtersAria}>
        {FILTERS.map((filter) => (
          <FilterPill
            key={filter.id}
            icon={filter.icon}
            label={copy.filters[filter.id]}
            ariaLabelPrefix={copy.filterAriaPrefix}
            active={activeFilters.has(filter.id)}
            onToggle={() => toggleFilter(filter.id)}
          />
        ))}
      </div>

      {loading ? (
        <p className="results-hint">{copy.getSearchingText(query)}</p>
      ) : results && results.length > 0 ? (
        <>
          <p className="results-count">{copy.getResultsCountText(totalResults, query)}</p>

          <div className="results-list">
            {results.slice((page - 1) * pageSize, page * pageSize).map((item) => (
              <article className="result-item" key={item.doc_id}>
                <div className="result-left">
                  <div className="result-icon">
                    {getFileTypeIcon(item.file_type)}
                  </div>
                </div>

                <div className="result-body">
                  <a href={item.url} target="_blank" rel="noreferrer" className="result-title">{item.title}</a>
                  <p className="result-snippet">{item.snippet}</p>
                  <div className="result-meta">{item.domain} • {item.crawl_date ? item.crawl_date.split("T")[0] : ""}</div>
                </div>

                <div className="result-right">
                  <span className="result-tag">{getLocalizedFileType(item.file_type, copy)}</span>
                </div>
              </article>
            ))}
          </div>

          <div className="results-pager">
            <button aria-label={copy.previousPageAria} onClick={() => setPage(Math.max(1, page - 1))} disabled={page === 1}>{"<"}</button>
            <span className="pager-info">{page}</span>
            <button aria-label={copy.nextPageAria} onClick={() => setPage(page + 1)} disabled={page * pageSize >= results.length}>{">"}</button>
          </div>
        </>
      ) : (
        <p className="results-hint">{copy.emptyResults}</p>
      )}
    </section>
  );
}

export default SearchPortal;
