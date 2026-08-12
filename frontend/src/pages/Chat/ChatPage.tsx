import { ComingSoonPanel } from "../../components/ComingSoonPanel";
import { PageIntro } from "../../components/PageIntro";

export function ChatPage() {
  return (
    <div>
      <PageIntro
        eyebrow="Ask with evidence"
        title="Chat"
        description="Explore a collection through questions backed by retrieved source chunks."
      />
      <ComingSoonPanel
        step="Next capability"
        title="Grounded chat coming soon."
        description="The future chat experience will make context and citations inspectable. No API calls, conversation state, or answer generation are connected yet."
      />
    </div>
  );
}
