function FilterPill({ icon, label, ariaLabelPrefix, active = false, onToggle }) {
  return (
    <button
      type="button"
      className={`filter-pill ${active ? "is-active" : ""}`}
      aria-pressed={active}
      onClick={onToggle}
      aria-label={`${ariaLabelPrefix} ${label}`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

export default FilterPill;
