"use client";

import { createContext, useContext, useEffect, useState } from "react";

export type DockBehavior = "auto-hide" | "shrink";

interface UIPreferences {
  dockBehavior: DockBehavior;
  setDockBehavior: (behavior: DockBehavior) => void;
}

const UIPreferencesContext = createContext<UIPreferences | undefined>(undefined);

export function useUIPreferences() {
  const context = useContext(UIPreferencesContext);
  if (!context) {
    throw new Error("useUIPreferences must be used within a UIPreferencesProvider");
  }
  return context;
}

export function UIPreferencesProvider({ children }: { children: React.ReactNode }) {
  const [dockBehavior, setDockBehaviorState] = useState<DockBehavior>("shrink");
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("aether-ui-dock-behavior");
    if (stored === "auto-hide" || stored === "shrink") {
      setDockBehaviorState(stored);
    }
    setIsLoaded(true);
  }, []);

  const setDockBehavior = (behavior: DockBehavior) => {
    setDockBehaviorState(behavior);
    localStorage.setItem("aether-ui-dock-behavior", behavior);
  };

  if (!isLoaded) {
    // Return children immediately to prevent hydration mismatch/layout shifts if possible,
    // or render null if strict sync is needed. Here we render children with default "shrink".
    // Since this affects a fixed UI element (Dock), a slight shift is acceptable or we can just render.
    // For now, render children to avoid blocking the whole app.
  }

  return (
    <UIPreferencesContext.Provider value={{ dockBehavior, setDockBehavior }}>
      {children}
    </UIPreferencesContext.Provider>
  );
}
