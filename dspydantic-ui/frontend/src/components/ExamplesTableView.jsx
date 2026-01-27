import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { examplesApi, tasksApi } from '../services/api'
import ExampleModal from './ExampleModal'
import ExampleFilters from './ExampleFilters'

export default function ExamplesTableView({ taskId, task }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [filters, setFilters] = useState({
    input_complete: null,
    output_complete: null,
    status: null,
    created_after: '',
    created_before: '',
  })
  const [page, setPage] = useState(0)
  const [editingExample, setEditingExample] = useState(null)
  const [showDuplicateModal, setShowDuplicateModal] = useState(false)
  const [duplicateExampleId, setDuplicateExampleId] = useState(null)
  const [selectedTaskId, setSelectedTaskId] = useState('')
  const [showGenerateModal, setShowGenerateModal] = useState(false)
  const [generateExampleId, setGenerateExampleId] = useState(null)
  const [generateModelId, setGenerateModelId] = useState('')
  const [generateApiKey, setGenerateApiKey] = useState('')
  const [selectedExamples, setSelectedExamples] = useState(new Set())
  const [showBulkStatusModal, setShowBulkStatusModal] = useState(false)
  const [showBulkGenerateModal, setShowBulkGenerateModal] = useState(false)
  const [bulkStatus, setBulkStatus] = useState('')
  const [bulkGenerateModelId, setBulkGenerateModelId] = useState('')
  const [bulkGenerateApiKey, setBulkGenerateApiKey] = useState('')
  const pageSize = 50

  const { data: examples, isLoading } = useQuery({
    queryKey: ['examples', taskId, filters, page],
    queryFn: async () => {
      const params = {
        offset: page * pageSize,
        limit: pageSize,
      }
      
      // Add filters
      if (filters.input_complete !== null && filters.input_complete !== undefined) {
        params.input_complete = filters.input_complete === true || filters.input_complete === 'true'
      }
      if (filters.output_complete !== null && filters.output_complete !== undefined) {
        params.output_complete = filters.output_complete === true || filters.output_complete === 'true'
      }
      if (filters.status !== null && filters.status !== '') {
        params.status = filters.status
      }
      if (filters.created_after) {
        // Convert date to ISO format
        const date = new Date(filters.created_after)
        params.created_after = date.toISOString()
      }
      if (filters.created_before) {
        // Convert date to ISO format, set to end of day
        const date = new Date(filters.created_before)
        date.setHours(23, 59, 59, 999)
        params.created_before = date.toISOString()
      }
      
      const response = await examplesApi.list(taskId, params)
      return response.data
    },
  })

  const updateStatusMutation = useMutation({
    mutationFn: ({ id, status }) => examplesApi.updateStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries(['examples', taskId])
    },
  })

  const deleteMutation = useMutation({
    mutationFn: examplesApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries(['examples', taskId])
    },
  })

  const { data: allTasks } = useQuery({
    queryKey: ['tasks'],
    queryFn: async () => {
      const response = await tasksApi.list()
      return response.data
    },
  })

  const duplicateMutation = useMutation({
    mutationFn: ({ exampleId, newTaskId }) => examplesApi.duplicate(exampleId, { new_task_id: newTaskId }),
    onSuccess: () => {
      queryClient.invalidateQueries(['examples', taskId])
      setShowDuplicateModal(false)
      setDuplicateExampleId(null)
      setSelectedTaskId('')
    },
  })

  const generateOutputMutation = useMutation({
    mutationFn: ({ exampleId, modelId, apiKey }) => {
      const requestBody = {}
      if (modelId) requestBody.model_id = modelId
      if (apiKey) requestBody.api_key = apiKey
      // Always send at least an empty object to ensure FastAPI receives a valid request body
      return examplesApi.generateOutput(exampleId, requestBody)
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['examples', taskId])
      setShowGenerateModal(false)
      setGenerateExampleId(null)
      setGenerateModelId('')
      setGenerateApiKey('')
    },
  })

  const bulkUpdateStatusMutation = useMutation({
    mutationFn: ({ exampleIds, status }) => examplesApi.bulkUpdateStatus({ example_ids: exampleIds, status }),
    onSuccess: () => {
      queryClient.invalidateQueries(['examples', taskId])
      setSelectedExamples(new Set())
      setShowBulkStatusModal(false)
      setBulkStatus('')
    },
  })

  const bulkGenerateOutputMutation = useMutation({
    mutationFn: ({ exampleIds, modelId, apiKey }) => {
      const requestBody = { example_ids: exampleIds }
      if (modelId) requestBody.model_id = modelId
      if (apiKey) requestBody.api_key = apiKey
      return examplesApi.bulkGenerateOutput(requestBody)
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['examples', taskId])
      setSelectedExamples(new Set())
      setShowBulkGenerateModal(false)
      setBulkGenerateModelId('')
      setBulkGenerateApiKey('')
    },
  })

  const handleDuplicate = (exampleId) => {
    setDuplicateExampleId(exampleId)
    setShowDuplicateModal(true)
  }

  const handleDuplicateSubmit = (e) => {
    e.preventDefault()
    if (!selectedTaskId) return
    duplicateMutation.mutate({
      exampleId: duplicateExampleId,
      newTaskId: parseInt(selectedTaskId),
    })
  }

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }))
    setPage(0)
  }

  const handleStatusChange = (exampleId, status) => {
    updateStatusMutation.mutate({ id: exampleId, status })
  }

  const getStatusColor = (status) => {
    const colors = {
      approved: 'bg-green-100 text-green-800',
      rejected: 'bg-red-100 text-red-800',
      pending: 'bg-yellow-100 text-yellow-800',
      reviewed: 'bg-blue-100 text-blue-800',
    }
    return colors[status] || 'bg-gray-100 text-gray-800'
  }

  const getCompletenessBadge = (isComplete) => {
    return isComplete ? (
      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
        Complete
      </span>
    ) : (
      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
        Incomplete
      </span>
    )
  }

  if (isLoading) {
    return <div className="text-center py-8">Loading examples...</div>
  }

  const handleClearFilters = () => {
    setFilters({
      input_complete: null,
      output_complete: null,
      status: null,
      created_after: '',
      created_before: '',
    })
    setPage(0)
  }

  const handleGenerateOutput = (exampleId) => {
    setGenerateExampleId(exampleId)
    setGenerateModelId(task?.default_model || 'gpt-4o')
    setGenerateApiKey('')
    setShowGenerateModal(true)
  }

  const handleGenerateSubmit = (e) => {
    e.preventDefault()
    if (!generateExampleId) return
    // Prevent starting a new generation if one is already in progress
    if (generateOutputMutation.isLoading) return
    generateOutputMutation.mutate({
      exampleId: generateExampleId,
      modelId: generateModelId,
      apiKey: generateApiKey || undefined
    })
  }

  const handleSelectExample = (exampleId, checked) => {
    setSelectedExamples(prev => {
      const next = new Set(prev)
      if (checked) {
        next.add(exampleId)
      } else {
        next.delete(exampleId)
      }
      return next
    })
  }

  const handleSelectAll = (checked) => {
    if (checked) {
      setSelectedExamples(new Set(examples?.map(e => e.id) || []))
    } else {
      setSelectedExamples(new Set())
    }
  }

  const handleBulkStatusSubmit = (e) => {
    e.preventDefault()
    if (selectedExamples.size === 0) return
    bulkUpdateStatusMutation.mutate({
      exampleIds: Array.from(selectedExamples),
      status: bulkStatus || null
    })
  }

  const handleBulkGenerateSubmit = (e) => {
    e.preventDefault()
    if (selectedExamples.size === 0) return
    // Prevent starting a new generation if one is already in progress
    if (bulkGenerateOutputMutation.isLoading) return
    bulkGenerateOutputMutation.mutate({
      exampleIds: Array.from(selectedExamples),
      modelId: bulkGenerateModelId,
      apiKey: bulkGenerateApiKey || undefined
    })
  }

  return (
    <div>
      {/* Filters */}
      <ExampleFilters
        filters={filters}
        onFilterChange={handleFilterChange}
        onClearFilters={handleClearFilters}
      />

      {/* Bulk Actions Bar */}
      {selectedExamples.size > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4 flex items-center justify-between">
          <div className="text-sm font-medium text-blue-900">
            {selectedExamples.size} example{selectedExamples.size !== 1 ? 's' : ''} selected
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => {
                setBulkStatus('')
                setShowBulkStatusModal(true)
              }}
              className="px-3 py-1.5 text-sm font-medium text-blue-700 bg-blue-100 hover:bg-blue-200 rounded-md"
            >
              Change Status
            </button>
            <button
              onClick={() => {
                setBulkGenerateModelId(task?.default_model || 'gpt-4o')
                setBulkGenerateApiKey('')
                setShowBulkGenerateModal(true)
              }}
              className="px-3 py-1.5 text-sm font-medium text-purple-700 bg-purple-100 hover:bg-purple-200 rounded-md"
            >
              Generate Completions
            </button>
            <button
              onClick={() => setSelectedExamples(new Set())}
              className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 border border-gray-300 rounded-md"
            >
              Clear Selection
            </button>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  <input
                    type="checkbox"
                    checked={examples && examples.length > 0 && examples.every(e => selectedExamples.has(e.id))}
                    onChange={(e) => handleSelectAll(e.target.checked)}
                    onClick={(e) => e.stopPropagation()}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  ID
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Input Complete
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Output Complete
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {examples && examples.length > 0 ? (
                examples.map((example) => (
                  <tr
                    key={example.id}
                    className="hover:bg-gray-50 cursor-pointer"
                    onClick={() => navigate(`/tasks/${taskId}/examples/${example.id}`, { state: { filters } })}
                  >
                    <td className="px-6 py-4 whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={selectedExamples.has(example.id)}
                        onChange={(e) => handleSelectExample(example.id, e.target.checked)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      #{example.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {getCompletenessBadge(example.input_complete)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {getCompletenessBadge(example.output_complete)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm" onClick={(e) => e.stopPropagation()}>
                      <select
                        value={example.status || ''}
                        onChange={(e) =>
                          handleStatusChange(example.id, e.target.value || null)
                        }
                        className={`px-2 py-1 rounded text-xs font-medium border-0 ${getStatusColor(
                          example.status || ''
                        )}`}
                      >
                        <option value="">No Status</option>
                        <option value="approved">Approved</option>
                        <option value="rejected">Rejected</option>
                        <option value="pending">Pending</option>
                        <option value="reviewed">Reviewed</option>
                      </select>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(example.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex justify-end space-x-2" onClick={(e) => e.stopPropagation()}>
                        <Link
                          to={`/tasks/${taskId}/examples/${example.id}/view`}
                          state={{ filters }}
                          className="inline-flex items-center px-2 py-1 text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded"
                          title="View"
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
                        </Link>
                        <button
                          onClick={() => setEditingExample(example)}
                          className="inline-flex items-center px-2 py-1 text-blue-600 hover:text-blue-900 hover:bg-blue-50 rounded"
                          title="Quick Edit"
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
                              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                            />
                          </svg>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDuplicate(example.id)
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
                        {(!example.output_complete || Object.keys(example.output_data || {}).length === 0) && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleGenerateOutput(example.id)
                            }}
                            disabled={generateOutputMutation.isLoading}
                            className="inline-flex items-center px-2 py-1 text-purple-600 hover:text-purple-900 hover:bg-purple-50 rounded disabled:opacity-50"
                            title="Generate Output"
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
                                d="M13 10V3L4 14h7v7l9-11h-7z"
                              />
                            </svg>
                          </button>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            if (
                              window.confirm(
                                'Are you sure you want to delete this example?'
                              )
                            ) {
                              deleteMutation.mutate(example.id)
                            }
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
                    colSpan="7"
                    className="px-6 py-4 text-center text-sm text-gray-500"
                  >
                    No examples found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {examples && examples.length === pageSize && (
          <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
            <div className="flex-1 flex justify-between sm:hidden">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setPage((p) => p + 1)}
                className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
              >
                Next
              </button>
            </div>
            <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
              <div>
                <p className="text-sm text-gray-700">
                  Page <span className="font-medium">{page + 1}</span>
                </p>
              </div>
              <div>
                <nav
                  className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px"
                  aria-label="Pagination"
                >
                  <button
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                    disabled={page === 0}
                    className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => setPage((p) => p + 1)}
                    className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50"
                  >
                    Next
                  </button>
                </nav>
              </div>
            </div>
          </div>
        )}
      </div>

      {editingExample && task && (
        <ExampleModal
          taskId={taskId}
          task={task}
          example={editingExample}
          onClose={() => setEditingExample(null)}
          onSuccess={() => {
            setEditingExample(null)
            queryClient.invalidateQueries(['examples', taskId])
          }}
        />
      )}

      {showDuplicateModal && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Duplicate Example</h3>
            <form onSubmit={handleDuplicateSubmit}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Select Target Task
                </label>
                <select
                  value={selectedTaskId}
                  onChange={(e) => setSelectedTaskId(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  required
                >
                  <option value="">Select a task...</option>
                  {allTasks
                    ?.filter((t) => t.id !== taskId)
                    .map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name}
                      </option>
                    ))}
                </select>
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowDuplicateModal(false)
                    setDuplicateExampleId(null)
                    setSelectedTaskId('')
                  }}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={duplicateMutation.isLoading || !selectedTaskId}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
                >
                  {duplicateMutation.isLoading ? 'Duplicating...' : 'Duplicate'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showGenerateModal && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Generate Output</h3>
            <form onSubmit={handleGenerateSubmit}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model ID
                </label>
                <input
                  type="text"
                  value={generateModelId}
                  onChange={(e) => setGenerateModelId(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  placeholder="e.g., gpt-4o"
                  required
                />
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  API Key (optional, uses environment variable if not provided)
                </label>
                <input
                  type="password"
                  value={generateApiKey}
                  onChange={(e) => setGenerateApiKey(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Leave empty to use OPENAI_API_KEY"
                />
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowGenerateModal(false)
                    setGenerateExampleId(null)
                    setGenerateModelId('')
                    setGenerateApiKey('')
                  }}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={generateOutputMutation.isLoading || !generateModelId}
                  className="px-4 py-2 bg-purple-600 text-white rounded-md text-sm font-medium hover:bg-purple-500 disabled:opacity-50"
                >
                  {generateOutputMutation.isLoading ? 'Generating...' : 'Generate'}
                </button>
              </div>
            </form>
            {generateOutputMutation.isError && (
              <div className="mt-4 text-sm text-red-600">
                Error: {generateOutputMutation.error?.response?.data?.detail || 'Failed to generate output'}
              </div>
            )}
          </div>
        </div>
      )}

      {showBulkStatusModal && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Change Status for {selectedExamples.size} Example{selectedExamples.size !== 1 ? 's' : ''}
            </h3>
            <form onSubmit={handleBulkStatusSubmit}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Status
                </label>
                <select
                  value={bulkStatus}
                  onChange={(e) => setBulkStatus(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">No Status</option>
                  <option value="approved">Approved</option>
                  <option value="rejected">Rejected</option>
                  <option value="pending">Pending</option>
                  <option value="reviewed">Reviewed</option>
                </select>
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowBulkStatusModal(false)
                    setBulkStatus('')
                  }}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={bulkUpdateStatusMutation.isLoading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
                >
                  {bulkUpdateStatusMutation.isLoading ? 'Updating...' : 'Update Status'}
                </button>
              </div>
            </form>
            {bulkUpdateStatusMutation.isError && (
              <div className="mt-4 text-sm text-red-600">
                Error: {bulkUpdateStatusMutation.error?.response?.data?.detail || 'Failed to update status'}
              </div>
            )}
          </div>
        </div>
      )}

      {showBulkGenerateModal && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Generate Completions for {selectedExamples.size} Example{selectedExamples.size !== 1 ? 's' : ''}
            </h3>
            <form onSubmit={handleBulkGenerateSubmit}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model ID
                </label>
                <input
                  type="text"
                  value={bulkGenerateModelId}
                  onChange={(e) => setBulkGenerateModelId(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  placeholder="e.g., gpt-4o"
                  required
                />
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  API Key (optional, uses environment variable if not provided)
                </label>
                <input
                  type="password"
                  value={bulkGenerateApiKey}
                  onChange={(e) => setBulkGenerateApiKey(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Leave empty to use OPENAI_API_KEY"
                />
              </div>
              <div className="mb-4 text-sm text-gray-600">
                Generation will run in the background. Examples will be updated as they complete.
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowBulkGenerateModal(false)
                    setBulkGenerateModelId('')
                    setBulkGenerateApiKey('')
                  }}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={bulkGenerateOutputMutation.isLoading || !bulkGenerateModelId}
                  className="px-4 py-2 bg-purple-600 text-white rounded-md text-sm font-medium hover:bg-purple-500 disabled:opacity-50"
                >
                  {bulkGenerateOutputMutation.isLoading ? 'Starting...' : 'Start Generation'}
                </button>
              </div>
            </form>
            {bulkGenerateOutputMutation.isError && (
              <div className="mt-4 text-sm text-red-600">
                Error: {bulkGenerateOutputMutation.error?.response?.data?.detail || 'Failed to start generation'}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
