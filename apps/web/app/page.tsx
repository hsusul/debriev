"use client";

import { useEffect, useState } from "react";
import ChatPanel from "./components/ChatPanel";
import UploadModal from "./components/UploadModal";
import NewProjectButton from "./NewProjectButton";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type ProjectItem = {
  project_id: string;
  name: string;
};

export default function HomePage() {
  const [projects, setProjects] = useState<ProjectItem[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState("");

  useEffect(() => {
    let cancelled = false;
    const loadProjects = async () => {
      try {
        const response = await fetch(`${API_BASE}/v1/projects`);
        if (!response.ok || cancelled) {
          return;
        }
        const payload = (await response.json()) as ProjectItem[];
        if (cancelled) {
          return;
        }
        setProjects(payload);
        setSelectedProjectId((prev) => prev || payload[0]?.project_id || "");
      } catch {
        if (!cancelled) {
          setProjects([]);
        }
      }
    };

    void loadProjects();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="mx-auto flex min-h-[calc(100vh-10rem)] w-full max-w-5xl flex-col gap-8">
      <div className="mx-auto w-full max-w-3xl text-center">
        <div className="space-y-3">
          <h2 className="text-xl font-semibold text-slate-100">Workspace</h2>
          <p className="mx-auto max-w-xl text-sm text-slate-300">
            Upload legal PDFs, create projects, and iterate on verification reports in one place.
          </p>
        </div>
        <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row sm:items-stretch">
          <UploadModal projectId={selectedProjectId || undefined} />
          <NewProjectButton variant="card" />
          <div className="flex aspect-square w-40 flex-col justify-between rounded-2xl border border-slate-700/70 bg-slate-800/60 p-4 text-left">
            <div>
              <p className="text-sm font-semibold text-slate-100">Chat this project</p>
              <p className="mt-1 text-xs text-slate-400">Use combined retrieval across project documents.</p>
            </div>
            <select
              value={selectedProjectId}
              onChange={(event) => setSelectedProjectId(event.target.value)}
              className="h-9 w-full rounded-lg border border-slate-600 bg-slate-800 px-2 text-xs text-slate-100 focus:border-blue-400 focus:outline-none"
            >
              {projects.length === 0 ? <option value="">No projects</option> : null}
              {projects.map((project) => (
                <option key={project.project_id} value={project.project_id}>
                  {project.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <ChatPanel
        className="min-h-[20rem] flex-1"
        projectId={selectedProjectId || undefined}
        introText={
          selectedProjectId
            ? "Ask a question and I will answer from indexed chunks across this project."
            : "Ask a question and I will answer from indexed chunks."
        }
      />
    </section>
  );
}
