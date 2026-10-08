import { Link, Navigate, Route, Routes, useParams } from "react-router-dom";
import AppHeader from "./components/AppHeader";
import AnalysisProgress from "./pages/AnalysisProgress";
import AssessmentReport from "./pages/AssessmentReport";
import Assessments from "./pages/Assessments";
import AssessmentView from "./pages/AssessmentView";
import NewAssessment from "./pages/NewAssessment";
import QuestionnaireExample from "./pages/QuestionnaireExample";

function LegacyStatusRedirect() {
  const { id } = useParams();
  return <Navigate to={`/assessments/${id}/progress`} replace />;
}

export default function App() {
  return (
    <>
      <AppHeader />
      <main id="main">
        <Routes>
          <Route path="/" element={<NewAssessment />} />
          <Route path="/assessments" element={<Assessments />} />
          <Route path="/assessments/:id" element={<AssessmentView />} />
          <Route path="/assessments/:id/progress" element={<AnalysisProgress />} />
          <Route path="/assessments/:id/status" element={<LegacyStatusRedirect />} />
          <Route path="/assessments/:id/report" element={<AssessmentReport />} />
          <Route path="/questionnaire" element={<QuestionnaireExample />} />
          <Route path="/questionnaire-example" element={<Navigate to="/questionnaire" replace />} />
          <Route path="*" element={<div className="page"><p>Page not found. <Link to="/">Start a new assessment</Link></p></div>} />
        </Routes>
      </main>
    </>
  );
}
