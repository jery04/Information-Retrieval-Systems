function FilterPill({ icon, label, active = false, onToggle }) {
  return (
    <button
      type="button"
      className={`filter-pill ${active ? "is-active" : ""}`}
      aria-pressed={active}
      onClick={onToggle}
      aria-label={`Filtrar por ${label}`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

export default FilterPill;
