import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Network Intrusion Detection | Four-Model Demo",
  description: "Read-only interactive interface for frozen EECS 3404 model artifacts."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
