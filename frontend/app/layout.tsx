import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Regime Identification",
  description:
    "Live dashboard for the fair-evaluation regime-identification service.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-white text-neutral-900 antialiased">{children}</body>
    </html>
  );
}
