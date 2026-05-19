import React from 'react';
import './PillTabs.css';

interface Tab {
  id: string;
  label: string;
}

interface PillTabsProps {
  tabs: Tab[];
  activeTab: string;
  onChange: (id: string) => void;
}

export function PillTabs({ tabs, activeTab, onChange }: PillTabsProps) {
  return (
    <div className="pill-tabs-container">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`pill-tab ${activeTab === tab.id ? 'active' : ''}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
