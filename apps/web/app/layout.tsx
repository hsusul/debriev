import "./globals.css";
import { FileText, Folder, Link as LinkIcon } from "lucide-react";
import type { ReactNode } from "react";
import NewProjectButton from "./NewProjectButton";

const API_BASE = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://api:8000";

type ProjectItem = {
  project_id: string;
  name: string;
  created_at: string;
};

type DocumentItem = {
  doc_id: string;
  project_id?: string | null;
  filename: string;
  created_at: string;
};

async function loadProjects(): Promise<ProjectItem[]> {
  const response = await fetch(`${API_BASE}/v1/projects`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load projects");
  }
  return (await response.json()) as ProjectItem[];
}

async function loadDocuments(): Promise<DocumentItem[]> {
  const response = await fetch(`${API_BASE}/v1/documents`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load documents");
  }
  return (await response.json()) as DocumentItem[];
}

export const metadata = {
  title: "Debriev",
  description: "Debriev v1 stub"
};

export default async function RootLayout({ children }: { children: ReactNode }) {
  let projects: ProjectItem[] = [];
  let documents: DocumentItem[] = [];

  try {
    [projects, documents] = await Promise.all([loadProjects(), loadDocuments()]);
  } catch {
    projects = [];
    documents = [];
  }

  const docsByProject = new Map<string, DocumentItem[]>();
  const unassigned: DocumentItem[] = [];
  for (const doc of documents) {
    if (doc.project_id) {
      const current = docsByProject.get(doc.project_id) || [];
      current.push(doc);
      docsByProject.set(doc.project_id, current);
    } else {
      unassigned.push(doc);
    }
  }

  return (
    <html lang="en">
      <body>
        <div className="grid min-h-screen grid-cols-[17rem_minmax(0,1fr)] bg-slate-950">
          <aside className="sticky top-0 h-screen overflow-y-auto border-r border-slate-800 bg-slate-950 px-4 py-5 text-slate-100">
            <div className="mb-6 space-y-2">
              <a href="/" className="inline-flex items-center gap-2 text-lg font-semibold text-white hover:text-slate-200">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-slate-800 text-sm font-bold">
                  D
                </span>
                Debriev
              </a>
              <p className="text-xs text-slate-400">Citation extraction and report workflow.</p>
            </div>

            <div className="mb-6 rounded-lg border border-slate-800 bg-slate-900/60 p-2">
              <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">Quick links</p>
              <div className="flex items-center gap-1">
                <a href="/upload" className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-slate-300 hover:bg-slate-800">
                  <LinkIcon className="h-3.5 w-3.5 text-blue-300" aria-hidden="true" />
                  Upload
                </a>
                <NewProjectButton variant="sidebar" />
              </div>
            </div>

            <section>
              <h2 className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                <Folder className="h-3.5 w-3.5" aria-hidden="true" />
                Projects
              </h2>
              {projects.length === 0 ? <p className="text-xs text-slate-500">No projects yet.</p> : null}

              <div className="space-y-2">
                {projects.map((project) => {
                  const projectDocs = docsByProject.get(project.project_id) || [];
                  return (
                    <details key={project.project_id} className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900/60">
                      <summary className="cursor-pointer list-none px-3 py-2 text-sm font-medium text-slate-100 hover:bg-slate-800/80">
                        {project.name}
                      </summary>
                      {projectDocs.length === 0 ? (
                        <p className="px-3 pb-2 text-xs text-slate-500">No documents.</p>
                      ) : (
                        <ul className="space-y-1 px-2 pb-2">
                          {projectDocs.map((doc) => (
                            <li key={doc.doc_id}>
                              <a
                                href={`/reports/${doc.doc_id}`}
                                className="block truncate rounded-md px-2 py-1 text-sm text-slate-300 hover:bg-slate-800 hover:text-white"
                              >
                                {doc.filename}
                              </a>
                            </li>
                          ))}
                        </ul>
                      )}
                    </details>
                  );
                })}
              </div>
            </section>

            <section className="mt-6">
              <h2 className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
                <FileText className="h-3.5 w-3.5" aria-hidden="true" />
                Your Chats
              </h2>
              {unassigned.length === 0 ? (
                <p className="text-xs text-slate-500">No chats yet.</p>
              ) : (
                <ul className="space-y-1">
                  {unassigned.map((doc) => (
                    <li key={doc.doc_id}>
                      <a
                        href={`/reports/${doc.doc_id}`}
                        className="block truncate rounded-md px-2 py-1 text-sm text-slate-300 hover:bg-slate-800 hover:text-white"
                      >
                        {doc.filename}
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </aside>

          <div className="min-w-0 bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950">
            <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-900/85 backdrop-blur">
              <div className="flex h-16 items-center justify-between px-8">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Workspace</p>
                  <h1 className="text-base font-semibold text-slate-100">Debriev</h1>
                </div>
                <nav className="flex items-center gap-2 text-sm">
                  <a href="/about" className="rounded-md px-2 py-1 text-slate-300 hover:bg-slate-800 hover:text-slate-100">
                    About
                  </a>
                  <a href="/contact" className="rounded-md px-2 py-1 text-slate-300 hover:bg-slate-800 hover:text-slate-100">
                    Contact
                  </a>
                  <a href="/login" className="rounded-md px-2 py-1 text-slate-300 hover:bg-slate-800 hover:text-slate-100">
                    Login
                  </a>
                </nav>
              </div>
            </header>

            <main className="px-8 py-8">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
