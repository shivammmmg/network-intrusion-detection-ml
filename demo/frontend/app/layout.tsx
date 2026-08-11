import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Network Intrusion Detection | Four-Model Demo",
  description: "Interactive dashboard for finalized EECS 3404 model artifacts."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
