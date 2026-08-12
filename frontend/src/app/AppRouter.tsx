import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AppLayout } from "../layouts/AppLayout";
import { ChatPage } from "../pages/Chat/ChatPage";
import { DocumentsPage } from "../pages/Documents/DocumentsPage";
import { HomePage } from "../pages/Home/HomePage";
import { UploadPage } from "../pages/Upload/UploadPage";

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index element={<HomePage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="documents" element={<DocumentsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
