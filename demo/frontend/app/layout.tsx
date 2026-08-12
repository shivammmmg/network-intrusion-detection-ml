import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Network Intrusion Detection | Four-Model Demo",
  description: "Inference console for four finalized EECS 3404 intrusion-detection models trained on UNSW-NB15."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
