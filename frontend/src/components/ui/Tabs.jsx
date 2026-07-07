import { useState } from 'react';

export default function Tabs({ tabs, defaultTab = 0 }) {
  const [active, setActive] = useState(defaultTab);

  return (
    <div className="tabs-container">
      <div className="tabs-header">
        {tabs.map((tab, i) => (
          <button
            key={i}
            className={`tab-btn${i === active ? ' active' : ''}`}
            onClick={() => setActive(i)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="tab-content">
        {tabs[active] && tabs[active].content}
      </div>
    </div>
  );
}
