import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { evaluationApi } from '../services/api'
import EvaluationRunsTableView from './EvaluationRunsTableView'
import StartEvaluationModal from './StartEvaluationModal'

export default function EvaluationTab({ taskId, task }) {
  const [showStartModal, setShowStartModal] = useState(false)

  return (
    <div>
      <div className="sm:flex sm:items-center sm:justify-between mb-4">
        <div className="sm:flex-auto">
          <h2 className="text-lg font-medium text-gray-900">Evaluation Runs</h2>
          <p className="mt-1 text-sm text-gray-500">
            View and manage evaluation runs
          </p>
        </div>
        <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
          <button
            onClick={() => setShowStartModal(true)}
            className="inline-flex items-center justify-center rounded-md border border-transparent bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-500"
          >
            Start Evaluation
          </button>
        </div>
      </div>

      <EvaluationRunsTableView taskId={taskId} task={task} />

      {showStartModal && (
        <StartEvaluationModal
          taskId={taskId}
          task={task}
          onClose={() => setShowStartModal(false)}
          onSuccess={() => setShowStartModal(false)}
        />
      )}
    </div>
  )
}
