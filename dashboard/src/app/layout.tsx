import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import SessionSync from "@/components/SessionSync";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-jakarta",
});

export const metadata: Metadata = {
  title: "aether",
  description: "Your AI companion",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={jakarta.variable}>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="font-sans">
        <TooltipProvider>
          <SessionSync />
          <main className="app-scene">
            <div className="app-scene-bg app-scene-bg-animate" />
            <div className="app-scene-vignette app-scene-vignette-animate" />
            <div className="app-scene-content">
              <div className="app-glass-shell app-glass-shell-animate">
                <div className="app-shell-content">{children}</div>
              </div>
            </div>
          </main>
          <Toaster />
        </TooltipProvider>
      </body>
    </html>
  );
}
