import { useState, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { promptsApi, tasksApi } from '../services/api'
import PromptEditor from './PromptEditor'
import DiffViewer from './DiffViewer'

/**
 * Recursively extract all field paths from a schema
 */
function extractFieldPaths(schema, prefix = '') {
  const paths = []
  if (!schema || !schema.properties) return paths

  for (const [fieldName, fieldDef] of Object.entries(schema.properties)) {
    const fieldPath = prefix ? `${prefix}.${fieldName}` : fieldName
    paths.push(fieldPath)

    // Handle nested objects
    if (fieldDef.type === 'object' && fieldDef.properties) {
      paths.push(...extractFieldPaths(fieldDef, fieldPath))
    }

    // Handle arrays of objects
    if (fieldDef.type === 'array' && fieldDef.items && fieldDef.items.type === 'object' && fieldDef.items.properties) {
      paths.push(...extractFieldPaths(fieldDef.items, `${fieldPath}[]`))
    }
  }

  return paths
}

export default function PromptsTab({ taskId }) {
  const location = useLocation()
  const [selectedVersion, setSelectedVersion] = useState(null)
  const [compareVersion, setCompareVersion] = useState(null)
  const [isEditing, setIsEditing] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [deleteInfo, setDeleteInfo] = useState(null)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const queryClient = useQueryClient()

  const { data: versions, isLoading, error } = useQuery({
    queryKey: ['prompts', taskId],
    queryFn: async () => {
      const response = await promptsApi.list(taskId)
      return response.data
    },
  })

  // Handle navigation from evaluation/optimization runs
  useEffect(() => {
    if (location.state?.selectedVersionId && versions && versions.length > 0) {
      const version = versions.find(v => v.id === location.state.selectedVersionId)
      if (version && (!selectedVersion || selectedVersion.id !== version.id)) {
        setSelectedVersion(version)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state?.selectedVersionId, versions])

  const { data: task } = useQuery({
    queryKey: ['tasks', taskId],
    queryFn: async () => {
      const response = await tasksApi.get(taskId)
      return response.data
    },
  })

  const { data: comparison, isLoading: isComparing } = useQuery({
    queryKey: ['prompts', 'compare', selectedVersion?.id, compareVersion?.id],
    queryFn: async () => {
      if (!selectedVersion || !compareVersion) return null
      const response = await promptsApi.compare(selectedVersion.id, compareVersion.id)
      return response.data
    },
    enabled: !!selectedVersion && !!compareVersion,
  })

  const activateMutation = useMutation({
    mutationFn: promptsApi.activate,
    onSuccess: () => {
      queryClient.invalidateQueries(['prompts', taskId])
    },
  })

  const createMutation = useMutation({
    mutationFn: (data) => promptsApi.create(taskId, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['prompts', taskId])
      setIsCreating(false)
      setIsEditing(false)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ versionId, data }) => promptsApi.update(versionId, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['prompts', taskId])
      setIsEditing(false)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: promptsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries(['prompts', taskId])
      queryClient.invalidateQueries(['evaluation-runs', taskId])
      setSelectedVersion(null)
      setShowDeleteModal(false)
      setDeleteInfo(null)
    },
  })

  const handleDeleteClick = async () => {
    if (!selectedVersion) return
    try {
      const response = await promptsApi.getDeleteInfo(selectedVersion.id)
      setDeleteInfo(response.data)
      setShowDeleteModal(true)
    } catch (error) {
      console.error('Error fetching delete info:', error)
    }
  }

  const handleConfirmDelete = () => {
    if (selectedVersion) {
      deleteMutation.mutate(selectedVersion.id)
    }
  }

  const handleSavePrompt = (promptData) => {
    if (isCreating) {
      createMutation.mutate(promptData)
    } else if (selectedVersion) {
      updateMutation.mutate({ versionId: selectedVersion.id, data: promptData })
    }
  }

  const handlePromptChange = (promptData) => {
    // This can be used for real-time preview if needed
  }

  if (isLoading) {
    return <div className="text-center py-8">Loading prompts...</div>
  }

  if (error) {
    return (
      <div className="text-center py-12 bg-white rounded-lg shadow">
        <p className="text-red-600 mb-2">Error loading prompts</p>
        <p className="text-sm text-gray-500">
          {error.response?.data?.detail || error.message || 'Unknown error'}
        </p>
      </div>
    )
  }

  const activeVersion = versions?.find((v) => v.is_active)
  const hasNoPrompts = !versions || versions.length === 0

  return (
    <div>
      {hasNoPrompts && (
        <div className="mb-4 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">
                No Prompt Version Created
              </h3>
              <div className="mt-2 text-sm text-yellow-700">
                <p>
                  This task does not have a prompt version yet. System Prompt and Instruction Prompt Template 
                  should be defined when creating the task, or you can create a prompt version manually.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
      <div className="sm:flex sm:items-center sm:justify-between mb-4">
        <div className="sm:flex-auto">
          <h2 className="text-lg font-medium text-gray-900">Prompt Versions</h2>
          <p className="mt-1 text-sm text-gray-500">
            Manage system prompts and instruction prompt templates
          </p>
        </div>
        <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
          <button
            type="button"
            onClick={() => {
              setIsCreating(true)
              setSelectedVersion(null)
              setCompareVersion(null)
            }}
            className="inline-flex items-center justify-center rounded-md border border-transparent bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-500"
          >
            + Create New Prompt
          </button>
        </div>
      </div>

      {isCreating && (
        <div className="mb-6 bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Create New Prompt Version</h3>
                <PromptEditor
                  systemPrompt={activeVersion?.system_prompt || ''}
                  instructionPrompt={activeVersion?.instruction_prompt || ''}
                  schema={task?.pydantic_schema}
                  outputSchemaDescriptions={activeVersion?.output_schema_descriptions}
                  inputSchema={task?.input_schema}
                  onChange={handlePromptChange}
                  onSave={handleSavePrompt}
                  onCancel={() => setIsCreating(false)}
                />
        </div>
      )}

      {versions && versions.length === 0 && !isCreating ? (
        <div className="text-center py-12 bg-white rounded-lg shadow">
          <p className="text-gray-500">No prompt versions yet.</p>
          <button
            type="button"
            onClick={() => setIsCreating(true)}
            className="mt-4 text-blue-600 hover:text-blue-900"
          >
            Create your first prompt version
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 bg-white shadow overflow-hidden sm:rounded-md">
            <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
              <h3 className="text-sm font-medium text-gray-900">Versions</h3>
            </div>
            <ul className="divide-y divide-gray-200 max-h-[600px] overflow-y-auto">
              {versions?.map((version) => (
                <li
                  key={version.id}
                  className={`px-4 py-3 cursor-pointer hover:bg-gray-50 ${
                    selectedVersion?.id === version.id ? 'bg-blue-50' : ''
                  }`}
                  onClick={() => {
                    setSelectedVersion(version)
                    setCompareVersion(null)
                    setIsEditing(false)
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center">
                        <span className="text-sm font-medium text-gray-900">
                          v{version.version_number}
                        </span>
                        {version.is_active && (
                          <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                            Active
                          </span>
                        )}
                      </div>
                      <div className="mt-1 text-xs text-gray-500">
                        {new Date(version.created_at).toLocaleString()}
                      </div>
                      {version.created_by && (
                        <div className="mt-1 text-xs text-gray-400">
                          by {version.created_by}
                        </div>
                      )}
                    </div>
                    {!version.is_active && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          activateMutation.mutate(version.id)
                        }}
                        className="ml-2 text-xs text-blue-600 hover:text-blue-900"
                      >
                        Activate
                      </button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="lg:col-span-2 space-y-6">
            {selectedVersion && !isEditing && (
              <>
                <div className="bg-white shadow rounded-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-medium text-gray-900">
                        Version {selectedVersion.version_number}
                      </h3>
                      {selectedVersion.parent_version_id && (
                        <p className="text-xs text-gray-500 mt-1">
                          Based on version {versions?.find((v) => v.id === selectedVersion.parent_version_id)?.version_number || 'N/A'}
                        </p>
                      )}
                    </div>
                    <div className="flex space-x-2">
                      <button
                        type="button"
                        onClick={() => setIsEditing(true)}
                        className="text-sm text-blue-600 hover:text-blue-900"
                      >
                        Edit
                      </button>
                      {!selectedVersion.is_active && (
                        <button
                          type="button"
                          onClick={handleDeleteClick}
                          className="text-sm text-red-600 hover:text-red-900"
                        >
                          Delete
                        </button>
                      )}
                      {versions && versions.length > 1 && (
                        <select
                          value={compareVersion?.id || ''}
                          onChange={(e) => {
                            const version = versions.find((v) => v.id === parseInt(e.target.value))
                            setCompareVersion(version || null)
                          }}
                          className="text-sm border border-gray-300 rounded-md px-2 py-1"
                        >
                          <option value="">Compare with...</option>
                          {versions
                            .filter((v) => v.id !== selectedVersion.id)
                            .map((v) => (
                              <option key={v.id} value={v.id}>
                                v{v.version_number}
                              </option>
                            ))}
                        </select>
                      )}
                    </div>
                  </div>

                  {!compareVersion ? (
                    <div className="space-y-4">
                      {selectedVersion.system_prompt && (
                        <div>
                          <h4 className="text-sm font-medium text-gray-700 mb-2">System Prompt</h4>
                          <div className="bg-gray-50 rounded-md p-4">
                            <pre className="text-sm text-gray-800 whitespace-pre-wrap font-mono">
                              {selectedVersion.system_prompt}
                            </pre>
                          </div>
                        </div>
                      )}

                      {selectedVersion.instruction_prompt && (
                        <div>
                          <h4 className="text-sm font-medium text-gray-700 mb-2">
                            Instruction Prompt Template
                          </h4>
                          <div className="bg-gray-50 rounded-md p-4">
                            <pre className="text-sm text-gray-800 whitespace-pre-wrap font-mono">
                              {selectedVersion.instruction_prompt}
                            </pre>
                          </div>
                        </div>
                      )}

                      {selectedVersion.prompt_content && !selectedVersion.system_prompt && !selectedVersion.instruction_prompt && (
                        <div>
                          <h4 className="text-sm font-medium text-gray-700 mb-2">Prompt Content</h4>
                          <div className="bg-gray-50 rounded-md p-4">
                            <pre className="text-sm text-gray-800 whitespace-pre-wrap font-mono">
                              {selectedVersion.prompt_content}
                            </pre>
                          </div>
                        </div>
                      )}

                      {task?.pydantic_schema && (
                        <div>
                          <h4 className="text-sm font-medium text-gray-700 mb-2">Output Schema Field Descriptions</h4>
                          <div className="bg-gray-50 rounded-md p-4">
                            {(() => {
                              const allFieldPaths = extractFieldPaths(task.pydantic_schema)
                              const descriptions = selectedVersion.output_schema_descriptions || {}
                              
                              if (allFieldPaths.length === 0) {
                                return (
                                  <div className="text-sm text-gray-500 italic">
                                    No fields found in the output schema.
                                  </div>
                                )
                              }
                              
                              return allFieldPaths.map((fieldPath) => {
                                const description = descriptions[fieldPath]
                                return (
                                  <div key={fieldPath} className="mb-3 last:mb-0">
                                    <div className="text-xs font-medium text-gray-700 mb-1">{fieldPath}</div>
                                    {description ? (
                                      <div className="text-sm text-gray-800 whitespace-pre-wrap">{description}</div>
                                    ) : (
                                      <div className="text-sm text-gray-400 italic">No description defined</div>
                                    )}
                                  </div>
                                )
                              })
                            })()}
                          </div>
                        </div>
                      )}

                      {selectedVersion.metrics && (
                        <div>
                          <h4 className="text-sm font-medium text-gray-700 mb-2">Metrics</h4>
                          <pre className="text-xs text-gray-600 bg-gray-50 p-3 rounded border">
                            {JSON.stringify(selectedVersion.metrics, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-2">System Prompt Diff</h4>
                        <DiffViewer
                          oldText={compareVersion.system_prompt || ''}
                          newText={selectedVersion.system_prompt || ''}
                        />
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-2">
                          Instruction Prompt Diff
                        </h4>
                        <DiffViewer
                          oldText={compareVersion.instruction_prompt || ''}
                          newText={selectedVersion.instruction_prompt || ''}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}

            {selectedVersion && isEditing && (
              <div className="bg-white shadow rounded-lg p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Edit Version {selectedVersion.version_number}
                </h3>
                <PromptEditor
                  systemPrompt={selectedVersion.system_prompt || ''}
                  instructionPrompt={selectedVersion.instruction_prompt || ''}
                  schema={task?.pydantic_schema}
                  outputSchemaDescriptions={selectedVersion.output_schema_descriptions}
                  inputSchema={task?.input_schema}
                  onChange={handlePromptChange}
                  onSave={handleSavePrompt}
                  onCancel={() => setIsEditing(false)}
                />
              </div>
            )}

            {!selectedVersion && !isCreating && (
              <div className="bg-white shadow rounded-lg p-12 text-center">
                <p className="text-gray-500">Select a version to view details</p>
              </div>
            )}
          </div>
        </div>
      )}

      {showDeleteModal && deleteInfo && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Delete Prompt Version</h3>
            <div className="mb-4">
              <p className="text-sm text-gray-700 mb-3">
                Are you sure you want to delete version {selectedVersion?.version_number}? This action cannot be undone.
              </p>
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
                <p className="text-sm font-medium text-yellow-800 mb-2">Warning: The following will be deleted:</p>
                <ul className="text-sm text-yellow-700 list-disc list-inside space-y-1">
                  <li>{deleteInfo.evaluation_runs_count} evaluation run{deleteInfo.evaluation_runs_count !== 1 ? 's' : ''} associated with this prompt version</li>
                  {deleteInfo.optimization_run_id && (
                    <li>The optimization run that created this version (ID: {deleteInfo.optimization_run_id})</li>
                  )}
                </ul>
              </div>
            </div>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => {
                  setShowDeleteModal(false)
                  setDeleteInfo(null)
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
    </div>
  )
}
