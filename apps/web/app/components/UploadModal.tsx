"use client";

import { Upload, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type ProjectItem = {
  project_id: string;
  name: string;
};

type UploadModalProps = {
  projectId?: string;
};

async function formatError(response: Response, fallback: string): Promise<string> {
  const raw = await response.text().catch(() => "");
  let detail = raw.trim();

  try {
    const json = JSON.parse(raw) as { detail?: string };
    if (json.detail) {
      detail = json.detail;
    }
  } catch {
    // Keep raw text fallback.
  }

  const suffix = detail ? ` - ${detail}` : "";
  return `${response.status} ${response.statusText || fallback}${suffix}`;
}

export default function UploadModal({ projectId }: UploadModalProps) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [projects, setProjects] = useState<ProjectItem[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState(projectId ?? "");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSelectedProjectId(projectId ?? "");
  }, [projectId]);

  useEffect(() => {
    if (!open) {
      return;
    }
    if (projectId) {
      return;
    }

    const loadProjects = async () => {
      try {
        const response = await fetch(`${API_BASE}/v1/projects`);
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as ProjectItem[];
        setProjects(payload);
      } catch {
        setProjects([]);
      }
    };

    void loadProjects();
  }, [open, projectId]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");

    if (!file) {
      setError("Select a PDF file.");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const resolvedProjectId = projectId || selectedProjectId;
      if (resolvedProjectId) {
        formData.append("project_id", resolvedProjectId);
      }

      const response = await fetch(`${API_BASE}/v1/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(await formatError(response, "Upload failed"));
      }

      const payload = (await response.json()) as { doc_id: string };
      setOpen(false);
      router.push(`/reports/${payload.doc_id}`);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Upload failed");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="group flex aspect-square w-40 flex-col items-start justify-between rounded-2xl border border-slate-700/70 bg-slate-800/60 p-5 text-left transition hover:border-blue-400/60 hover:bg-slate-800"
      >
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-blue-500/20 text-blue-300 group-hover:bg-blue-500/30">
          <Upload className="h-5 w-5" aria-hidden="true" />
        </span>
        <span className="space-y-1">
          <span className="block text-base font-semibold text-slate-100">Upload PDF</span>
          <span className="block text-xs text-slate-400">Send a new document for analysis.</span>
        </span>
      </button>

      {open ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl border border-slate-700 bg-slate-900 p-5 shadow-xl">
            <div className="mb-4 flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold text-slate-100">Upload PDF</h3>
                <p className="text-xs text-slate-400">Choose a file to create a report.</p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                aria-label="Close"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>

            <form onSubmit={onSubmit} className="space-y-4">
              {!projectId && projects.length > 0 ? (
                <div className="space-y-1">
                  <label htmlFor="modal-project" className="text-xs font-medium text-slate-300">
                    Project (optional)
                  </label>
                  <select
                    id="modal-project"
                    value={selectedProjectId}
                    onChange={(event) => setSelectedProjectId(event.target.value)}
                    className="h-10 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 text-sm text-slate-100 focus:border-blue-400 focus:outline-none"
                  >
                    <option value="">No project</option>
                    {projects.map((project) => (
                      <option key={project.project_id} value={project.project_id}>
                        {project.name}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}

              <div className="space-y-1">
                <label htmlFor="modal-file" className="text-xs font-medium text-slate-300">
                  PDF file
                </label>
                <input
                  id="modal-file"
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(event) => setFile(event.target.files?.[0] || null)}
                  className="block w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-100 file:mr-3 file:rounded-md file:border-0 file:bg-blue-500 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-white hover:file:bg-blue-400"
                />
              </div>

              {error ? (
                <p className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">{error}</p>
              ) : null}

              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  className="rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-200 hover:bg-slate-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-lg bg-blue-500 px-3 py-2 text-sm font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? "Uploading..." : "Upload"}
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : null}
    </>
  );
}
