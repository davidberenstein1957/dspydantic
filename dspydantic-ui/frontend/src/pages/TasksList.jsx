import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { tasksApi } from '../services/api'
import CreateTaskModal from '../components/CreateTaskModal'
import TaskFilters from '../components/TaskFilters'

export default function TasksList() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showDuplicateModal, setShowDuplicateModal] = useState(false)
  const [duplicateTaskId, setDuplicateTaskId] = useState(null)
  const [newTaskName, setNewTaskName] = useState('')
  const [copyExamples, setCopyExamples] = useState(true)
  const [sortField, setSortField] = useState('name')
  const [sortDirection, setSortDirection] = useState('asc')
  const [filters, setFilters] = useState({
    search: '',
    has_examples: null,
    created_after: '',
    created_before: '',
  })
  const queryClient = useQueryClient()
  const navigate = useNavigate()

  const { data: tasks, isLoading, isError, error } = useQuery({
    queryKey: ['tasks'],
    queryFn: async () => {
      const response = await tasksApi.list()
      return response.data
    },
    retry: 1,
  })

  const duplicateMutation = useMutation({
    mutationFn: ({ taskId, data }) => tasksApi.duplicate(taskId, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['tasks'])
      setShowDuplicateModal(false)
      setDuplicateTaskId(null)
      setNewTaskName('')
      setCopyExamples(true)
    },
  })

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
      search: '',
      has_examples: null,
      created_after: '',
      created_before: '',
    })
  }

  const handleDuplicate = (taskId, taskName, e) => {
    e.stopPropagation()
    setDuplicateTaskId(taskId)
    setNewTaskName(`${taskName} (Copy)`)
    setShowDuplicateModal(true)
  }

  const handleDuplicateSubmit = (e) => {
    e.preventDefault()
    if (!newTaskName.trim()) return
    duplicateMutation.mutate({
      taskId: duplicateTaskId,
      data: {
        new_name: newTaskName.trim(),
        copy_examples: copyExamples,
      },
    })
  }

  const filteredAndSortedTasks = useMemo(() => {
    if (!tasks || !Array.isArray(tasks)) return []

    let filtered = tasks.filter((task) => {
      // Search filter
      if (filters.search) {
        const searchText = filters.search.toLowerCase()
        const matchesName = task.name.toLowerCase().includes(searchText)
        const matchesDescription = task.description && task.description.toLowerCase().includes(searchText)
        if (!matchesName && !matchesDescription) return false
      }

      // Has examples filter
      if (filters.has_examples !== null) {
        const hasExamples = task.example_count > 0
        if (hasExamples !== filters.has_examples) return false
      }

      // Created after filter
      if (filters.created_after) {
        const taskDate = new Date(task.created_at)
        const filterDate = new Date(filters.created_after)
        if (taskDate < filterDate) return false
      }

      // Created before filter
      if (filters.created_before) {
        const taskDate = new Date(task.created_at)
        const filterDate = new Date(filters.created_before)
        filterDate.setHours(23, 59, 59, 999)
        if (taskDate > filterDate) return false
      }

      return true
    })

    filtered.sort((a, b) => {
      let aVal, bVal

      switch (sortField) {
        case 'name':
          aVal = a.name.toLowerCase()
          bVal = b.name.toLowerCase()
          break
        case 'examples':
          aVal = a.example_count
          bVal = b.example_count
          break
        case 'completed_examples':
          aVal = a.completed_examples_count || 0
          bVal = b.completed_examples_count || 0
          break
        case 'optimization_score':
          aVal = a.last_optimization_score ?? -Infinity
          bVal = b.last_optimization_score ?? -Infinity
          break
        case 'last_update':
          aVal = a.last_prompt_update ? new Date(a.last_prompt_update).getTime() : 0
          bVal = b.last_prompt_update ? new Date(b.last_prompt_update).getTime() : 0
          break
        case 'created':
          aVal = new Date(a.created_at).getTime()
          bVal = new Date(b.created_at).getTime()
          break
        default:
          aVal = a.name.toLowerCase()
          bVal = b.name.toLowerCase()
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1
      return 0
    })

    return filtered
  }, [tasks, filters, sortField, sortDirection])

  const formatDate = (dateString) => {
    if (!dateString) return '-'
    const date = new Date(dateString)
    return date.toLocaleDateString()
  }

  const formatRelativeDate = (dateString) => {
    if (!dateString) return '-'
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  const SortIcon = ({ field }) => {
    if (sortField !== field) return <span className="text-gray-400">↕</span>
    return sortDirection === 'asc' ? <span>↑</span> : <span>↓</span>
  }

  if (isLoading) {
    return <div className="text-center py-12">Loading tasks...</div>
  }

  if (isError) {
    return (
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="text-center py-12">
          <p className="text-red-600 mb-4">
            Failed to load tasks: {error?.response?.data?.detail || error?.message || 'Unknown error'}
          </p>
          <button
            onClick={() => queryClient.invalidateQueries(['tasks'])}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-500"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="px-4 sm:px-6 lg:px-8">
      <div className="sm:flex sm:items-center mb-6">
        <div className="sm:flex-auto">
          <h1 className="text-2xl font-semibold text-gray-900">Tasks</h1>
          <p className="mt-2 text-sm text-gray-700">
            Manage your DSPy optimization tasks
          </p>
        </div>
        <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
          <button
            onClick={() => setShowCreateModal(true)}
            className="block rounded-md bg-blue-600 px-3 py-2 text-center text-sm font-semibold text-white shadow-sm hover:bg-blue-500"
          >
            Create Task
          </button>
        </div>
      </div>

      {tasks && tasks.length > 0 && (
        <>
          {/* Filters */}
          <TaskFilters
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
                      onClick={() => handleSort('name')}
                    >
                      <div className="flex items-center gap-2">
                        Name
                        <SortIcon field="name" />
                      </div>
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Description
                    </th>
                    <th
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('examples')}
                    >
                      <div className="flex items-center gap-2">
                        Examples
                        <SortIcon field="examples" />
                      </div>
                    </th>
                    <th
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('completed_examples')}
                    >
                      <div className="flex items-center gap-2">
                        Completed
                        <SortIcon field="completed_examples" />
                      </div>
                    </th>
                    <th
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('optimization_score')}
                    >
                      <div className="flex items-center gap-2">
                        Last Score
                        <SortIcon field="optimization_score" />
                      </div>
                    </th>
                    <th
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('last_update')}
                    >
                      <div className="flex items-center gap-2">
                        Last Update
                        <SortIcon field="last_update" />
                      </div>
                    </th>
                    <th
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                      onClick={() => handleSort('created')}
                    >
                      <div className="flex items-center gap-2">
                        Created
                        <SortIcon field="created" />
                      </div>
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredAndSortedTasks && filteredAndSortedTasks.length > 0 ? (
                    filteredAndSortedTasks.map((task) => (
                      <tr
                        key={task.id}
                        className="hover:bg-gray-50 cursor-pointer"
                        onClick={() => navigate(`/tasks/${task.id}`)}
                      >
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          <span className="text-blue-600 hover:text-blue-900">
                            {task.name}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-500">
                          <div className="max-w-xs truncate">
                            {task.description || '-'}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {task.example_count}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {task.completed_examples_count || 0}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {task.last_optimization_score !== null && task.last_optimization_score !== undefined
                            ? task.last_optimization_score.toFixed(3)
                            : '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatRelativeDate(task.last_prompt_update)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatDate(task.created_at)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                          <div className="flex justify-end space-x-2">
                            <button
                              onClick={(e) => handleDuplicate(task.id, task.name, e)}
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
                          </div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td
                        colSpan="8"
                        className="px-6 py-4 text-center text-sm text-gray-500"
                      >
                        No tasks found
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {tasks && tasks.length === 0 && (
        <div className="mt-8 text-center py-12 bg-white rounded-lg shadow">
          <p className="text-gray-500">No tasks yet. Create your first task to get started.</p>
        </div>
      )}

      {showCreateModal && (
        <CreateTaskModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            setShowCreateModal(false)
            queryClient.invalidateQueries(['tasks'])
          }}
        />
      )}

      {showDuplicateModal && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Duplicate Task</h3>
            <form onSubmit={handleDuplicateSubmit}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  New Task Name
                </label>
                <input
                  type="text"
                  value={newTaskName}
                  onChange={(e) => setNewTaskName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>
              <div className="mb-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={copyExamples}
                    onChange={(e) => setCopyExamples(e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Copy examples</span>
                </label>
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => {
                    setShowDuplicateModal(false)
                    setDuplicateTaskId(null)
                    setNewTaskName('')
                    setCopyExamples(true)
                  }}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={duplicateMutation.isLoading || !newTaskName.trim()}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
                >
                  {duplicateMutation.isLoading ? 'Duplicating...' : 'Duplicate'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
