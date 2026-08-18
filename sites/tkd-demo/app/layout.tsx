import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TKD Layout-Aware RAG Demo",
  description: "Layout-aware RAG demo over the TKD datasheet corpus, served from a static public snapshot.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
