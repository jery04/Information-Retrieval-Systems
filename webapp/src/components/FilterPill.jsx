function FilterPill({ icon, label }) {
  return (
    <button type="button" className="filter-pill" aria-label={`Filtrar por ${label}`}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

export default FilterPill;
