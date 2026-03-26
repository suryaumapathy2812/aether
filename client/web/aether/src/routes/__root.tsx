import { Suspense } from "react";
import { HeadContent, Scripts, createRootRouteWithContext, Navigate } from "@tanstack/react-router";
import { TooltipProvider } from "#/components/ui/tooltip";
import { Toaster } from "#/components/ui/sonner";
import ThemeProvider from "#/components/ThemeProvider";
import SessionSync from "#/components/SessionSync";
import NotificationProvider from "#/components/NotificationProvider";
import FloatingToolbar from "#/components/FloatingToolbar";
import KeyboardShortcutsProvider from "#/components/KeyboardShortcutsProvider";
import { UIPreferencesProvider } from "#/lib/ui-preferences";
import appCss from "../styles.css?url";

export const Route = createRootRouteWithContext<{}>()({
  notFoundComponent: () => <Navigate to="/login" />,
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      {
        name: "viewport",
        content:
          "width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover",
      },
      { name: "theme-color", content: "#0a0a0a" },
      { title: "aether" },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      { rel: "apple-touch-icon", href: "/app-icon-dark.svg" },
      { rel: "icon", type: "image/svg+xml", href: "/app-icon-dark.svg" },
    ],
  }),
  shellComponent: RootDocument,
});

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <HeadContent />
      </head>
      <body className="font-sans antialiased">
        <ThemeProvider>
          <UIPreferencesProvider>
            <TooltipProvider>
              <SessionSync />
              <NotificationProvider>
                <KeyboardShortcutsProvider>
                  <div className="h-dvh flex flex-col overflow-hidden">
                    <Suspense>
                      <FloatingToolbar />
                    </Suspense>
                    <main id="app-main" className="flex-1 overflow-hidden">
                      {children}
                    </main>
                  </div>
                </KeyboardShortcutsProvider>
              </NotificationProvider>
              <Toaster />
            </TooltipProvider>
          </UIPreferencesProvider>
        </ThemeProvider>
        <Scripts />
      </body>
    </html>
  );
}
