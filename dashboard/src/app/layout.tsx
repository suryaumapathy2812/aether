import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import SessionSync from "@/components/SessionSync";
import NotificationProvider from "@/components/NotificationProvider";
import ServiceWorkerRegistrar from "@/components/ServiceWorkerRegistrar";
import Sidebar from "@/components/Sidebar";
import KeyboardShortcutsProvider from "@/components/KeyboardShortcutsProvider";
import { UIPreferencesProvider } from "@/lib/ui-preferences";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-jakarta",
});

export const metadata: Metadata = {
  title: "aether",
  description: "Your AI companion",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Aether",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={jakarta.variable}>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover" />
        <meta name="theme-color" content="#0a0a0a" />
        <link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="192x192" href="/icons/icon-192.png" />
        <link rel="icon" type="image/png" sizes="512x512" href="/icons/icon-512.png" />
      </head>
      <body className="font-sans">
        <UIPreferencesProvider>
          <TooltipProvider>
            <SessionSync />
            <ServiceWorkerRegistrar />
            <NotificationProvider>
              <KeyboardShortcutsProvider>
                <div className="app-layout">
                  <Sidebar />
                  <main className="flex-1 min-w-0 h-full overflow-hidden">
                    {children}
                  </main>
                </div>
              </KeyboardShortcutsProvider>
            </NotificationProvider>
            <Toaster />
          </TooltipProvider>
        </UIPreferencesProvider>
      </body>
    </html>
  );
}
