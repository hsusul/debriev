import type { ReactNode } from "react";

export const metadata = {
  title: "Debriev",
  description: "Debriev v1 stub"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "sans-serif", padding: "24px" }}>
        <h1>Debriev</h1>
        <nav style={{ marginBottom: "16px" }}>
          <a href="/upload">Upload</a> | <a href="/documents">Documents</a>
        </nav>
        {children}
      </body>
    </html>
  );
}
