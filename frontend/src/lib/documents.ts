import type { JobInfo, LocationInfo } from "../api/types";

/** "folder / path" for a place a file was found, using forward slashes either way. */
export function locationPath(location: Pick<LocationInfo, "folder" | "path">): string {
  const folder = location.folder.replace(/[\\/]+$/, "");
  const path = location.path.replace(/\\/g, "/").replace(/^\/+/, "");
  return path ? `${folder}/${path}` : folder;
}

/** What to tell the reader about a location: present, or how long it has been gone. */
export function locationState(location: Pick<LocationInfo, "status" | "misses">): string {
  if (location.status === "present") return "found";
  return location.misses === 1 ? "not found in the last update" : `not found in the last ${location.misses} updates`;
}

type JobCounts = Pick<JobInfo, "state" | "message" | "added" | "replaced" | "missing" | "failed_files" | "indexed">;

/** One sentence on what a finished library update did, leaving out what stayed at zero. */
export function jobSummary(job: JobCounts): string {
  if (job.state === "running") return "running now";
  if (job.state === "interrupted") return "stopped before it finished; the next update carries on";
  if (job.state === "failed") return job.message ?? "stopped with an error";
  const parts: Array<[number, string]> = [
    [job.added, "added"],
    [job.replaced, "replaced by an edit"],
    [job.missing, "file not found"],
    [job.failed_files, "could not be read"],
    [job.indexed, "made searchable"],
  ];
  const shown = parts.filter(([count]) => count > 0).map(([count, label]) => `${count} ${label}`);
  return shown.length > 0 ? shown.join(", ") : "nothing changed";
}
