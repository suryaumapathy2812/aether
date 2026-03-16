"use client";

import { createContext, useContext } from "react";

interface UIPreferences {}

const UIPreferencesContext = createContext<UIPreferences | undefined>(undefined);

export function useUIPreferences() {
  const context = useContext(UIPreferencesContext);
  if (!context) {
    throw new Error("useUIPreferences must be used within a UIPreferencesProvider");
  }
  return context;
}

export function UIPreferencesProvider({ children }: { children: React.ReactNode }) {
  return (
    <UIPreferencesContext.Provider value={{}}>
      {children}
    </UIPreferencesContext.Provider>
  );
}
