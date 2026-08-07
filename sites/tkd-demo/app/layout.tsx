import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TKD Evidence RAG Demo",
  description: "Original TKD Evidence RAG design demo using a static public corpus snapshot.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}
