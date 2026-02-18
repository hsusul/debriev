"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type ProjectItem = {
  project_id: string;
  name: string;
};

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [projects, setProjects] = useState<ProjectItem[]>([]);
  const [projectId, setProjectId] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
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
  }, []);

  const formatUploadError = async (response: Response, fallback: string): Promise<string> => {
    const raw = await response.text().catch(() => "");
    let detail = raw.trim();

    try {
      const payload = JSON.parse(raw) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // Keep raw response text if not JSON.
    }

    const suffix = detail ? ` - ${detail}` : "";
    return `${response.status} ${response.statusText || fallback}${suffix}`;
  };

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
      if (projectId) {
        formData.append("project_id", projectId);
      }

      const response = await fetch(`${API_BASE}/v1/upload`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(await formatUploadError(response, "Upload failed"));
      }

      const payload = (await response.json()) as { doc_id: string };
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
    <main>
      <h2>Upload</h2>
      <form onSubmit={onSubmit}>
        <div style={{ marginBottom: "10px" }}>
          <label htmlFor="project">Project (optional)</label>
          <br />
          <select
            id="project"
            value={projectId}
            onChange={(event) => setProjectId(event.target.value)}
            style={{ minWidth: "260px" }}
          >
            <option value="">No project</option>
            {projects.map((project) => (
              <option key={project.project_id} value={project.project_id}>
                {project.name}
              </option>
            ))}
          </select>
        </div>

        <input
          type="file"
          accept="application/pdf,.pdf"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
        <button type="submit" disabled={loading} style={{ marginLeft: "8px" }}>
          {loading ? "Uploading..." : "Upload"}
        </button>
      </form>
      {error ? <p style={{ color: "crimson" }}>{error}</p> : null}
    </main>
  );
}
