import { ComingSoonPanel } from "../../components/ComingSoonPanel";
import { PageIntro } from "../../components/PageIntro";

export function DocumentsPage() {
  return (
    <div>
      <PageIntro
        eyebrow="Collection"
        title="Documents"
        description="Review ingested sources and the metadata that makes retrieval results traceable."
      />
      <ComingSoonPanel
        step="Next capability"
        title="Document browser coming soon."
        description="This foundation reserves a clear place for source inventory and document details. It does not read, filter, or delete backend data yet."
      />
    </div>
  );
}
