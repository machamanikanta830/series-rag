import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { AppRouter } from "./app/AppRouter";
import "./index.css";

const rootElement = document.getElementById("root");

if (rootElement === null) {
  throw new Error("SeriesRAG could not find the application root element.");
}

createRoot(rootElement).render(
  <StrictMode>
    <AppRouter />
  </StrictMode>,
);
