import type { Metadata } from "next";
import SessionSync from "@/components/SessionSync";
import "./globals.css";

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
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap"
          rel="stylesheet"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className="min-h-screen flex justify-center">
        <SessionSync />
        <div className="w-full max-w-[430px]">{children}</div>
      </body>
    </html>
  );
}
