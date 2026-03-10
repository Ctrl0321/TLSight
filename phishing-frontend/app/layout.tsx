import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "TLS Phishing Detector",
  description: "Detect phishing URLs using TLS + hostname features",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
