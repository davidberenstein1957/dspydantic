import { useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ErrorProvider, useError } from './contexts/ErrorContext'
import { setupErrorInterceptor } from './services/api'
import Layout from './components/Layout'
import ErrorNotification from './components/ErrorNotification'
import TasksList from './pages/TasksList'
import TaskDetail from './pages/TaskDetail'
import ExampleDetail from './pages/ExampleDetail'
import ExampleView from './pages/ExampleView'
import EvaluationResultsView from './pages/EvaluationResultsView'

const queryClient = new QueryClient()

function AppContent() {
  const { error, showError, dismissError } = useError()

  // Update error handler whenever showError changes
  useEffect(() => {
    setupErrorInterceptor(showError)
  }, [showError])

  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<TasksList />} />
          <Route path="/tasks" element={<TasksList />} />
          <Route path="/tasks/:id" element={<TaskDetail />} />
          <Route path="/tasks/:taskId/examples/:exampleId" element={<ExampleDetail />} />
          <Route path="/tasks/:taskId/examples/:exampleId/view" element={<ExampleView />} />
          <Route path="/tasks/:taskId/evaluations/:runId/results" element={<EvaluationResultsView />} />
        </Routes>
      </Layout>
      <ErrorNotification error={error} onDismiss={dismissError} />
    </Router>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorProvider>
        <AppContent />
      </ErrorProvider>
    </QueryClientProvider>
  )
}

export default App
