import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { PackageAllocationPage } from "./pages/PackageAllocationPage";
import { ReductionAnalyticsPage } from "./pages/ReductionAnalyticsPage";
import { FairnessV2Page } from "./pages/FairnessV2Page";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<PackageAllocationPage />} />
        <Route path="/reductions" element={<ReductionAnalyticsPage />} />
        <Route path="/clusters" element={<FairnessV2Page />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
