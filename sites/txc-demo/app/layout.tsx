import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";
import "./globals.css";
import { siteConfig } from "./site-config";

const sans = Manrope({ variable: "--font-sans", subsets: ["latin"] });
const mono = IBM_Plex_Mono({ variable: "--font-mono", subsets: ["latin"], weight: ["400", "500", "600"] });

export const metadata: Metadata = {
  title: `${siteConfig.corpus} Evidence RAG — Traceable Datasheet Search`,
  description: `A technical prototype demonstrating page-native, traceable RAG over ${siteConfig.corpus} engineering datasheets.`,
  icons: { icon: "/favicon.svg" },
  openGraph: {
    title: `${siteConfig.corpus} Evidence RAG`,
    description: "A technical prototype for page-native retrieval with evidence you can inspect.",
    images: ["/og.png"],
  },
  twitter: { card: "summary_large_image", images: ["/og.png"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${sans.variable} ${mono.variable}`}>{children}</body></html>;
}
