import { useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { examplesApi, tasksApi } from '../services/api'
import SchemaForm from './SchemaForm'
import ExampleModal from './ExampleModal'
import ExamplesTableView from './ExamplesTableView'

export default function ExamplesTab({ taskId }) {
  const [viewMode, setViewMode] = useState('table') // 'list' or 'table'
  const [showAddModal, setShowAddModal] = useState(false)
  const [editingExample, setEditingExample] = useState(null)
  const [uploadStatus, setUploadStatus] = useState(null)
  const fileInputRef = useRef(null)
  const queryClient = useQueryClient()

  const { data: task } = useQuery({
    queryKey: ['tasks', taskId],
    queryFn: async () => {
      const response = await tasksApi.get(taskId)
      return response.data
    },
  })

  const { data: examples, isLoading } = useQuery({
    queryKey: ['examples', taskId],
    queryFn: async () => {
      const response = await examplesApi.list(taskId)
      return response.data
    },
  })

  const { data: validationData } = useQuery({
    queryKey: ['schema-validation', taskId],
    queryFn: async () => {
      const response = await tasksApi.validateSchema(taskId)
      return response.data
    },
  })

  const deleteMutation = useMutation({
    mutationFn: examplesApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries(['examples', taskId])
      queryClient.invalidateQueries(['tasks', taskId])
    },
  })

  const uploadMutation = useMutation({
    mutationFn: (file) => examplesApi.uploadFile(taskId, file),
    onSuccess: (data) => {
      setUploadStatus({
        type: 'success',
        message: `Successfully imported ${data.data.created} of ${data.data.total} examples.`,
        errors: data.data.errors,
      })
      queryClient.invalidateQueries(['examples', taskId])
      queryClient.invalidateQueries(['tasks', taskId])
      // Clear status after 5 seconds
      setTimeout(() => setUploadStatus(null), 5000)
    },
    onError: (error) => {
      setUploadStatus({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to upload file',
      })
      setTimeout(() => setUploadStatus(null), 5000)
    },
  })

  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      const fileExt = file.name.toLowerCase().split('.').pop()
      if (!['csv', 'xlsx', 'xls'].includes(fileExt)) {
        setUploadStatus({
          type: 'error',
          message: 'Please select a CSV or Excel file (.csv, .xlsx, .xls)',
        })
        setTimeout(() => setUploadStatus(null), 5000)
        return
      }
      uploadMutation.mutate(file)
    }
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  if (isLoading) {
    return <div className="text-center py-8">Loading examples...</div>
  }

  return (
    <div>
      <div className="sm:flex sm:items-center sm:justify-between mb-4">
        <div className="sm:flex-auto">
          <h2 className="text-lg font-medium text-gray-900">Labeled Examples</h2>
          <p className="mt-1 text-sm text-gray-500">
            Manage training examples with input and output data
          </p>
        </div>
        <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none flex items-center space-x-3">
          <div className="flex rounded-md shadow-sm">
            <button
              onClick={() => setViewMode('table')}
              className={`px-3 py-2 text-sm font-medium rounded-l-md ${
                viewMode === 'table'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              Table
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`px-3 py-2 text-sm font-medium rounded-r-md ${
                viewMode === 'list'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              }`}
            >
              List
            </button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileSelect}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadMutation.isLoading}
            className="block rounded-md bg-green-600 px-3 py-2 text-center text-sm font-semibold text-white shadow-sm hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploadMutation.isLoading ? 'Uploading...' : 'Upload CSV/Excel'}
          </button>
          <button
            onClick={() => setShowAddModal(true)}
            className="block rounded-md bg-blue-600 px-3 py-2 text-center text-sm font-semibold text-white shadow-sm hover:bg-blue-500"
          >
            Add Example
          </button>
        </div>
      </div>

      {uploadStatus && (
        <div
          className={`mb-4 rounded-lg p-4 ${
            uploadStatus.type === 'success'
              ? 'bg-green-50 border border-green-200'
              : 'bg-red-50 border border-red-200'
          }`}
        >
          <div className="flex">
            <div className="flex-shrink-0">
              {uploadStatus.type === 'success' ? (
                <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : (
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
            </div>
            <div className="ml-3 flex-1">
              <p
                className={`text-sm font-medium ${
                  uploadStatus.type === 'success' ? 'text-green-800' : 'text-red-800'
                }`}
              >
                {uploadStatus.message}
              </p>
              {uploadStatus.errors && uploadStatus.errors.length > 0 && (
                <div className="mt-2 text-sm text-red-700">
                  <p className="font-medium">Errors:</p>
                  <ul className="list-disc list-inside mt-1">
                    {uploadStatus.errors.slice(0, 5).map((error, idx) => (
                      <li key={idx}>
                        Row {error.index + 1}: {error.error}
                      </li>
                    ))}
                    {uploadStatus.errors.length > 5 && (
                      <li>... and {uploadStatus.errors.length - 5} more errors</li>
                    )}
                  </ul>
                </div>
              )}
            </div>
            <div className="ml-auto pl-3">
              <button
                onClick={() => setUploadStatus(null)}
                className={`inline-flex rounded-md p-1.5 ${
                  uploadStatus.type === 'success'
                    ? 'text-green-500 hover:bg-green-100'
                    : 'text-red-500 hover:bg-red-100'
                }`}
              >
                <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}

      {viewMode === 'table' ? (
        task && <ExamplesTableView taskId={taskId} task={task} />
      ) : (
        <>
          {examples && examples.length === 0 ? (
            <div className="text-center py-12 bg-white rounded-lg shadow">
              <p className="text-gray-500">No examples yet. Add your first example to get started.</p>
            </div>
          ) : (
            <>
              {validationData && validationData.invalid_examples > 0 && (
                <div className="mb-4 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-yellow-800">
                        Schema Validation Issues
                      </h3>
                      <div className="mt-2 text-sm text-yellow-700">
                        <p>
                          {validationData.invalid_examples} of {validationData.total_examples} examples are missing required fields.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div className="bg-white shadow overflow-hidden sm:rounded-md">
                <ul className="divide-y divide-gray-200">
                  {examples?.map((example) => {
                    const validation = validationData?.validation_results?.find(
                      (v) => v.example_id === example.id
                    )
                    const hasIssues = validation && !validation.is_valid

                    return (
                      <li
                        key={example.id}
                        className={`px-6 py-4 ${hasIssues ? 'bg-yellow-50' : ''}`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            {hasIssues && (
                              <div className="mb-3 p-2 bg-yellow-100 border border-yellow-300 rounded text-xs">
                                <div className="font-medium text-yellow-800 mb-1">Missing Fields:</div>
                                {validation.missing_required_fields.length > 0 && (
                                  <div className="text-red-700">
                                    Required: {validation.missing_required_fields.join(', ')}
                                  </div>
                                )}
                                {validation.missing_optional_fields.length > 0 && (
                                  <div className="text-yellow-700">
                                    Optional: {validation.missing_optional_fields.join(', ')}
                                  </div>
                                )}
                              </div>
                            )}
                            <div className="text-sm">
                              <div className="font-medium text-gray-900">Example #{example.id}</div>
                              <div className="mt-2 text-xs text-gray-500">
                                {example.input_data?.images && Array.isArray(example.input_data.images) && (
                                  <span className="inline-flex items-center px-2 py-1 rounded bg-blue-50 text-blue-700 mr-2">
                                    {example.input_data.images.length} image{example.input_data.images.length > 1 ? 's' : ''}
                                  </span>
                                )}
                                {example.input_data?.text && (
                                  <span className="inline-flex items-center px-2 py-1 rounded bg-gray-50 text-gray-700">
                                    Text input
                                  </span>
                                )}
                              </div>
                              <div className="mt-2 text-xs text-gray-600">
                                <div className="font-medium mb-1">Input Preview:</div>
                                <pre className="text-xs text-gray-600 bg-gray-50 p-2 rounded max-h-24 overflow-auto">
                                  {example.input_data?.text 
                                    ? (example.input_data.text.length > 100 
                                        ? example.input_data.text.substring(0, 100) + '...' 
                                        : example.input_data.text)
                                    : example.input_data?.images 
                                      ? `${example.input_data.images.length} image(s)`
                                      : JSON.stringify(example.input_data, null, 2).substring(0, 100) + '...'}
                                </pre>
                              </div>
                            </div>
                          </div>
                          <div className="ml-4 flex flex-col space-y-2">
                            <Link
                              to={`/tasks/${taskId}/examples/${example.id}`}
                              className="inline-flex items-center text-blue-600 hover:text-blue-900 text-sm font-medium"
                            >
                              <svg
                                className="w-4 h-4 mr-1"
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
                              View
                            </Link>
                            <button
                              onClick={() => setEditingExample(example)}
                              className="inline-flex items-center text-blue-600 hover:text-blue-900 text-sm text-left"
                            >
                              <svg
                                className="w-4 h-4 mr-1"
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
                              Quick Edit
                            </button>
                            <button
                              onClick={() => {
                                if (window.confirm('Are you sure you want to delete this example?')) {
                                  deleteMutation.mutate(example.id)
                                }
                              }}
                              className="inline-flex items-center text-red-600 hover:text-red-900 text-sm text-left"
                            >
                              <svg
                                className="w-4 h-4 mr-1"
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
                              Delete
                            </button>
                          </div>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              </div>
            </>
          )}
        </>
      )}

      {showAddModal && task && (
        <ExampleModal
          taskId={taskId}
          task={task}
          onClose={() => setShowAddModal(false)}
          onSuccess={() => {
            setShowAddModal(false)
            queryClient.invalidateQueries(['examples', taskId])
            queryClient.invalidateQueries(['tasks', taskId])
          }}
        />
      )}

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
    </div>
  )
}
