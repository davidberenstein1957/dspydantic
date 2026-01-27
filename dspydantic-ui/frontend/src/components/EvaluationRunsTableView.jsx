import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { evaluationApi } from '../services/api'
import EvaluationRunFilters from './EvaluationRunFilters'
import StartEvaluationModal from './StartEvaluationModal'

export default function EvaluationRunsTableView({ taskId, task }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [sortField, setSortField] = useState('started_at')
  const [sortDirection, setSortDirection] = useState('desc')
  const [filters, setFilters] = useState({
    status: null,
    started_after: '',
    started_before: '',
    metric: null,
  })
  const [errorModal, setErrorModal] = useState({ isOpen: false, message: '' })
  const [showDuplicateModal, setShowDuplicateModal] = useState(false)
  const [duplicateRun, setDuplicateRun] = useState(null)

  const { data: runs, isLoading } = useQuery({
    queryKey: ['evaluation-runs', taskId],
    queryFn: async () => {
      const response = await evaluationApi.list(taskId)
      return response.data
    },
    refetchInterval: (query) => {
      const data = query.state.data
      if (!data || !Array.isArray(data)) return false
      const hasActiveRun = data.some((run) => ['pending', 'running'].includes(run.status))
      return hasActiveRun ? 2000 : false
    },
  })

  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [runToDelete, setRunToDelete] = useState(null)

  const deleteMutation = useMutation({
    mutationFn: evaluationApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries(['evaluation-runs', taskId])
      setShowDeleteModal(false)
      setRunToDelete(null)
    },
  })

  const filteredAndSortedRuns = useMemo(() => {
    if (!runs || !Array.isArray(runs)) return []
    
    let filtered = [...runs]
    
    // Apply filters
    if (filters.status) {
      filtered = filtered.filter(run => run.status === filters.status)
    }
    if (filters.started_after) {
      const date = new Date(filters.started_after)
      filtered = filtered.filter(run => {
        if (!run.started_at) return false
        return new Date(run.started_at) >= date
      })
    }
    if (filters.started_before) {
      const date = new Date(filters.started_before)
      date.setHours(23, 59, 59, 999)
      filtered = filtered.filter(run => {
        if (!run.started_at) return false
        return new Date(run.started_at) <= date
      })
    }
    if (filters.metric) {
      filtered = filtered.filter(run => run.config?.metric === filters.metric)
    }
    
    // Apply sorting
    return filtered.sort((a, b) => {
      let aVal, bVal

      switch (sortField) {
        case 'started_at':
          aVal = a.started_at ? new Date(a.started_at).getTime() : 0
          bVal = b.started_at ? new Date(b.started_at).getTime() : 0
          break
        case 'status':
          aVal = a.status
          bVal = b.status
          break
        case 'average_score':
          aVal = a.metrics?.average_score ?? -Infinity
          bVal = b.metrics?.average_score ?? -Infinity
          break
        case 'total_examples':
          aVal = a.metrics?.total_examples ?? 0
          bVal = b.metrics?.total_examples ?? 0
          break
        default:
          aVal = a.started_at ? new Date(a.started_at).getTime() : 0
          bVal = b.started_at ? new Date(b.started_at).getTime() : 0
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1
      return 0
    })
  }, [runs, filters, sortField, sortDirection])

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }))
  }

  const handleClearFilters = () => {
    setFilters({
      status: null,
      started_after: '',
      started_before: '',
      metric: null,
    })
  }

  const handleDuplicate = (run) => {
    setDuplicateRun(run)
    setShowDuplicateModal(true)
  }

  const handleDeleteClick = (run) => {
    setRunToDelete(run)
    setShowDeleteModal(true)
  }

  const handleConfirmDelete = () => {
    if (runToDelete) {
      deleteMutation.mutate(runToDelete.id)
    }
  }

  const SortIcon = ({ field }) => {
    if (sortField !== field) return <span className="text-gray-400">↕</span>
    return sortDirection === 'asc' ? <span>↑</span> : <span>↓</span>
  }

  const formatDate = (dateString) => {
    if (!dateString) return '-'
    return new Date(dateString).toLocaleString()
  }

  if (isLoading) {
    return <div className="text-center py-8">Loading evaluation runs...</div>
  }

  return (
    <div>
      {/* Filters */}
      <EvaluationRunFilters
        filters={filters}
        onFilterChange={handleFilterChange}
        onClearFilters={handleClearFilters}
      />

      {/* Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('started_at')}
                >
                  <div className="flex items-center gap-2">
                    Started
                    <SortIcon field="started_at" />
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('status')}
                >
                  <div className="flex items-center gap-2">
                    Status
                    <SortIcon field="status" />
                  </div>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Configuration
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('average_score')}
                >
                  <div className="flex items-center gap-2">
                    Average Score
                    <SortIcon field="average_score" />
                  </div>
                </th>
                <th
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('total_examples')}
                >
                  <div className="flex items-center gap-2">
                    Examples
                    <SortIcon field="total_examples" />
                  </div>
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredAndSortedRuns && filteredAndSortedRuns.length > 0 ? (
                filteredAndSortedRuns.map((run) => (
                  <tr
                    key={run.id}
                    className="hover:bg-gray-50 cursor-pointer"
                    onClick={() => {
                      if (run.status === 'completed') {
                        navigate(`/tasks/${taskId}/evaluations/${run.id}/results`)
                      } else if (run.error_message) {
                        setErrorModal({ isOpen: true, message: run.error_message })
                      }
                    }}
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(run.started_at)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          run.status === 'completed'
                            ? 'bg-green-100 text-green-800'
                            : run.status === 'running'
                            ? 'bg-blue-100 text-blue-800'
                            : run.status === 'failed'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {run.status}
                      </span>
                      {run.status === 'running' && (
                        <div className="mt-1 w-full bg-gray-200 rounded-full h-1">
                          <div
                            className="bg-blue-600 h-1 rounded-full"
                            style={{ width: `${(run.progress || 0) * 100}%` }}
                          />
                        </div>
                      )}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      <div className="flex flex-wrap gap-1">
                        {run.config?.model_id && (
                          <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                            Model: {run.config.model_id}
                          </span>
                        )}
                        {run.config?.metric && (
                          <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                            {run.config.metric}
                          </span>
                        )}
                        {run.prompt_version_id && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              navigate(`/tasks/${taskId}`, { state: { activeTab: 'prompts', selectedVersionId: run.prompt_version_id } })
                            }}
                            className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded hover:bg-blue-100"
                          >
                            Prompt v{run.prompt_version_number ?? run.prompt_version_id}
                          </button>
                        )}
                        {run.config?.max_examples && (
                          <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                            Max: {run.config.max_examples}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {run.metrics?.average_score !== undefined
                        ? `${(run.metrics.average_score * 100).toFixed(2)}%`
                        : '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {run.metrics?.total_examples || '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center justify-end gap-2" onClick={(e) => e.stopPropagation()}>
                        {run.status === 'completed' && (
                          <button
                            onClick={() => navigate(`/tasks/${taskId}/evaluations/${run.id}/results`)}
                            className="inline-flex items-center px-2 py-1 text-blue-600 hover:text-blue-900 hover:bg-blue-50 rounded"
                            title="View Results"
                          >
                            <svg
                              className="w-4 h-4"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                              />
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                              />
                            </svg>
                          </button>
                        )}
                        {run.error_message && (
                          <button
                            className="inline-flex items-center px-2 py-1 text-red-600 hover:text-red-900 hover:bg-red-50 rounded"
                            title="View Error"
                            onClick={(e) => {
                              e.stopPropagation()
                              setErrorModal({ isOpen: true, message: run.error_message })
                            }}
                          >
                            <svg
                              className="w-4 h-4"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                              />
                            </svg>
                          </button>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDuplicate(run)
                          }}
                          className="inline-flex items-center px-2 py-1 text-green-600 hover:text-green-900 hover:bg-green-50 rounded"
                          title="Duplicate"
                        >
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                            />
                          </svg>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDeleteClick(run)
                          }}
                          className="inline-flex items-center px-2 py-1 text-red-600 hover:text-red-900 hover:bg-red-50 rounded"
                          title="Delete"
                        >
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td
                    colSpan="6"
                    className="px-6 py-4 text-center text-sm text-gray-500"
                  >
                    No evaluation runs found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Error Modal */}
      {errorModal.isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden flex flex-col">
            <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">Error Details</h3>
              <button
                onClick={() => setErrorModal({ isOpen: false, message: '' })}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="px-6 py-4 overflow-y-auto flex-1">
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <pre className="text-sm text-red-800 whitespace-pre-wrap font-mono">{errorModal.message}</pre>
              </div>
            </div>
            <div className="px-6 py-4 border-t border-gray-200 flex justify-end">
              <button
                onClick={() => setErrorModal({ isOpen: false, message: '' })}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Delete Evaluation Run</h3>
            <div className="mb-4">
              <p className="text-sm text-gray-700 mb-3">
                Are you sure you want to delete this evaluation run? This action cannot be undone.
              </p>
            </div>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => {
                  setShowDeleteModal(false)
                  setRunToDelete(null)
                }}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmDelete}
                disabled={deleteMutation.isLoading}
                className="px-4 py-2 bg-red-600 text-white rounded-md text-sm font-medium hover:bg-red-500 disabled:opacity-50"
              >
                {deleteMutation.isLoading ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Duplicate Modal */}
      {showDuplicateModal && duplicateRun && task && (
        <StartEvaluationModal
          taskId={taskId}
          task={task}
          initialConfig={duplicateRun.config}
          onClose={() => {
            setShowDuplicateModal(false)
            setDuplicateRun(null)
          }}
          onSuccess={() => {
            setShowDuplicateModal(false)
            setDuplicateRun(null)
            queryClient.invalidateQueries(['evaluation-runs', taskId])
          }}
        />
      )}
    </div>
  )
}
