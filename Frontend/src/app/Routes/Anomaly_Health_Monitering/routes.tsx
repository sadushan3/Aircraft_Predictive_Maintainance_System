import { Navigate, Outlet, createHashRouter } from 'react-router-dom'

import Layout from '../../components/Layout/Anomaly_Health_Monitering/Layout'
import AnomalyMonitoring from '../../pages/Anomaly_Health_Monitering/AnomalyMonitoring'
import ConfidenceUncertainty from '../../pages/Anomaly_Health_Monitering/ConfidenceUncertainty'
import EngineDetail from '../../pages/Anomaly_Health_Monitering/EngineDetail'
import Explainability from '../../pages/Anomaly_Health_Monitering/Explainability'
import FeedbackLearning from '../../pages/Anomaly_Health_Monitering/FeedbackLearning'
import HealthMonitoring from '../../pages/Anomaly_Health_Monitering/HealthMonitoring'
import Overview from '../../pages/Anomaly_Health_Monitering/Overview'
import PipelineControl from '../../pages/Anomaly_Health_Monitering/PipelineControl'
import Reports from '../../pages/Anomaly_Health_Monitering/Reports'
import RootCause from '../../pages/Anomaly_Health_Monitering/RootCause'

function DashboardShell() {
  return (
    <Layout>
      <Outlet />
    </Layout>
  )
}

export const router = createHashRouter([
  {
    element: <DashboardShell />,
    children: [
      { path: '/', element: <Navigate to="/overview" replace /> },
      { path: '/overview', element: <Overview /> },
      { path: '/engine', element: <EngineDetail /> },
      { path: '/anomaly', element: <AnomalyMonitoring /> },
      { path: '/health', element: <HealthMonitoring /> },
      { path: '/rootcause', element: <RootCause /> },
      { path: '/confidence', element: <ConfidenceUncertainty /> },
      { path: '/explain', element: <Explainability /> },
      { path: '/feedback', element: <FeedbackLearning /> },
      { path: '/pipeline', element: <PipelineControl /> },
      { path: '/reports', element: <Reports /> },
      { path: '*', element: <Navigate to="/overview" replace /> },
    ],
  },
])
