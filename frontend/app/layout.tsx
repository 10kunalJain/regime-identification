import type { Metadata } from "next";
import { JetBrains_Mono, Newsreader } from "next/font/google";

import "./globals.css";
import { Providers } from "./providers";

const serif = Newsreader({
  subsets: ["latin"],
  variable: "--font-serif",
  weight: ["300", "400", "500", "600"],
  style: ["normal", "italic"],
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Regime Identification — Live Panel",
  description:
    "Filtered-only fair-evaluation benchmark for US-equity regime identification.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${serif.variable} ${mono.variable}`}>
      <body className="bg-paper font-serif text-ink antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
