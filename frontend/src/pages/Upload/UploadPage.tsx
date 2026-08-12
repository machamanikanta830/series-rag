import { ComingSoonPanel } from "../../components/ComingSoonPanel";
import { PageIntro } from "../../components/PageIntro";

export function UploadPage() {
  return (
    <div>
      <PageIntro
        eyebrow="Sources"
        title="Upload"
        description="Add source material to a collection while keeping its identity and provenance visible."
      />
      <ComingSoonPanel
        step="Next capability"
        title="Document upload coming soon."
        description="This page is a presentation shell only. File selection, validation, progress, and backend ingestion will be added during API integration."
      />
    </div>
  );
}
