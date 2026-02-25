import { readFile } from "node:fs/promises";
import { join } from "node:path";

const workerPath = join(process.cwd(), "node_modules", "pdfjs-dist", "build", "pdf.worker.min.mjs");

export async function GET(): Promise<Response> {
  const source = await readFile(workerPath, "utf8");
  return new Response(source, {
    headers: {
      "Content-Type": "text/javascript; charset=utf-8",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}
