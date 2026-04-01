import TopNav from "./components/TopNav";
import SearchPortal from "./components/SearchPortal";

function App() {
  return (
    <div className="app-root">
      <div className="page-card" role="application" aria-label="Portal de busqueda avanzada">
        <TopNav />

        {/* Contenido central de la pantalla de prueba. */}
        <main className="page-content">
          <SearchPortal />
        </main>
      </div>
    </div>
  );
}

export default App;
