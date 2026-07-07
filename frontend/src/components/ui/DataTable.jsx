export default function DataTable({ columns, rows, title }) {
  if (!rows || rows.length === 0) return null;

  return (
    <div>
      {title && <h4 className="section-header">{title}</h4>}
      <div className="data-table-wrapper">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key}>{col.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => (
                  <td key={col.key}>
                    {col.format ? col.format(row[col.key]) : row[col.key] ?? '—'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
