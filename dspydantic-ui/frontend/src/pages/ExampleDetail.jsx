import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { examplesApi, tasksApi } from '../services/api'
import ContentViewer from '../components/ContentViewer'
import SchemaForm from '../components/SchemaForm'
import ExampleFilters from '../components/ExampleFilters'

// Status options for keyboard navigation
const statusOptions = ['', 'approved', 'rejected', 'pending', 'reviewed']

export default function ExampleDetail() {
  const { taskId, exampleId } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const [outputData, setOutputData] = useState(null)
  const [showFilters, setShowFilters] = useState(false)
  // Load split position from localStorage per task, default to 50
  const [splitPosition, setSplitPosition] = useState(() => {
    if (taskId) {
      const cached = localStorage.getItem(`splitPosition_task_${taskId}`)
      return cached ? parseFloat(cached) : 50
    }
    return 50
  })
  const [isResizing, setIsResizing] = useState(false)
  const containerRef = useRef(null)
  const [selectedStatusIndex, setSelectedStatusIndex] = useState(-1)
  
  // Get filter params from location state or initialize empty
  const [filterParams, setFilterParams] = useState(location.state?.filters || {
    input_complete: null,
    output_complete: null,
    status: null,
    created_after: '',
    created_before: '',
  })
  const [showConfigureModelModal, setShowConfigureModelModal] = useState(false)

  const { data: task, isLoading: taskLoading } = useQuery({
    queryKey: ['tasks', taskId],
    queryFn: async () => {
      const response = await tasksApi.get(taskId)
      return response.data
    },
    enabled: !!taskId,
  })

  const { data: example, isLoading: exampleLoading } = useQuery({
    queryKey: ['examples', exampleId],
    queryFn: async () => {
      const response = await examplesApi.get(exampleId)
      return response.data
    },
    enabled: !!exampleId,
  })

  // Get list of examples for navigation
  const { data: examplesList } = useQuery({
    queryKey: ['examples', taskId, filterParams],
    queryFn: async () => {
      const params = {
        limit: 1000, // Get enough for navigation
      }
      
      // Add filters
      if (filterParams.input_complete !== null && filterParams.input_complete !== undefined) {
        params.input_complete = filterParams.input_complete === true || filterParams.input_complete === 'true'
      }
      if (filterParams.output_complete !== null && filterParams.output_complete !== undefined) {
        params.output_complete = filterParams.output_complete === true || filterParams.output_complete === 'true'
      }
      if (filterParams.status !== null && filterParams.status !== '') {
        params.status = filterParams.status
      }
      if (filterParams.created_after) {
        const date = new Date(filterParams.created_after)
        params.created_after = date.toISOString()
      }
      if (filterParams.created_before) {
        const date = new Date(filterParams.created_before)
        date.setHours(23, 59, 59, 999)
        params.created_before = date.toISOString()
      }
      
      const response = await examplesApi.list(taskId, params)
      return response.data
    },
    enabled: !!taskId,
  })

  // Calculate navigation indices
  const currentIndex = examplesList
    ? examplesList.findIndex((ex) => ex.id === parseInt(exampleId))
    : -1
  const hasPrevious = currentIndex > 0
  const hasNext = currentIndex >= 0 && currentIndex < (examplesList?.length || 0) - 1
  const previousExample = hasPrevious && examplesList ? examplesList[currentIndex - 1] : null
  const nextExample = hasNext && examplesList ? examplesList[currentIndex + 1] : null

  const handleFilterChange = (key, value) => {
    const newFilters = { ...filterParams, [key]: value }
    setFilterParams(newFilters)
    // Invalidate queries to refetch with new filters
    queryClient.invalidateQueries(['examples', taskId])
  }

  const handleClearFilters = () => {
    const clearedFilters = {
      input_complete: null,
      output_complete: null,
      status: null,
      created_after: '',
      created_before: '',
    }
    setFilterParams(clearedFilters)
    queryClient.invalidateQueries(['examples', taskId])
  }

  // Load split position when taskId changes
  useEffect(() => {
    if (taskId) {
      const cached = localStorage.getItem(`splitPosition_task_${taskId}`)
      if (cached) {
        setSplitPosition(parseFloat(cached))
      }
    }
  }, [taskId])

  // Save split position to localStorage whenever it changes
  useEffect(() => {
    if (taskId) {
      localStorage.setItem(`splitPosition_task_${taskId}`, splitPosition.toString())
    }
  }, [splitPosition, taskId])

  // Handle resizing
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const newPosition = ((e.clientX - rect.left) / rect.width) * 100
      setSplitPosition(Math.max(20, Math.min(80, newPosition)))
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
      }
    }
  }, [isResizing])

  // Navigate to first example when filters change and current example is not in filtered list
  useEffect(() => {
    if (examplesList && examplesList.length > 0 && currentIndex === -1 && exampleId) {
      // Current example not in filtered list, navigate to first
      const firstExampleId = examplesList[0].id
      if (parseInt(exampleId) !== firstExampleId) {
        navigate(`/tasks/${taskId}/examples/${firstExampleId}`, {
          state: { filters: filterParams },
          replace: true,
        })
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [examplesList, currentIndex])

  // Reset selected status index when example changes
  useEffect(() => {
    if (example?.status !== undefined) {
      const currentStatusIndex = statusOptions.indexOf(example.status || '')
      setSelectedStatusIndex(currentStatusIndex >= 0 ? currentStatusIndex : 0)
    }
  }, [exampleId, example?.status])

  const updateMutation = useMutation({
    mutationFn: async (data) => {
      const response = await examplesApi.update(exampleId, data)
      return response.data
    },
    onSuccess: (updatedExample) => {
      // Update outputData to reflect saved state
      if (updatedExample?.output_data) {
        setOutputData(updatedExample.output_data)
      }
      queryClient.invalidateQueries(['examples', exampleId])
      queryClient.invalidateQueries(['examples', taskId])
      queryClient.invalidateQueries(['tasks', taskId])
    },
  })

  const updateStatusMutation = useMutation({
    mutationFn: async (status) => {
      const response = await examplesApi.updateStatus(exampleId, status)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries(['examples', exampleId])
      queryClient.invalidateQueries(['examples', taskId])
    },
  })

  const generateOutputMutation = useMutation({
    mutationFn: async ({ modelId, apiKey }) => {
      const requestBody = {}
      if (modelId) requestBody.model_id = modelId
      if (apiKey) requestBody.api_key = apiKey
      // Always send at least an empty object to ensure FastAPI receives a valid request body
      const response = await examplesApi.generateOutput(exampleId, requestBody)
      return response.data
    },
    onSuccess: (updatedExample) => {
      if (updatedExample?.output_data) {
        setOutputData(updatedExample.output_data)
      }
      queryClient.invalidateQueries(['examples', exampleId])
      queryClient.invalidateQueries(['examples', taskId])
    },
  })

  // Handle save function
  const handleSave = useCallback(() => {
    if (outputData && example) {
      updateMutation.mutate({
        input_data: example.input_data,
        output_data: outputData,
      })
    }
  }, [outputData, example, updateMutation])

  // Handle generate function
  const handleGenerate = useCallback(() => {
    // Prevent starting a new generation if one is already in progress
    if (generateOutputMutation.isLoading) {
      return
    }
    if (task?.default_model) {
      generateOutputMutation.mutate({ 
        modelId: task.default_model, 
        apiKey: undefined 
      })
    } else {
      setShowConfigureModelModal(true)
    }
  }, [task?.default_model, generateOutputMutation])

  // Calculate hasChanges safely - computed early so it can be used in useEffect
  const hasChanges = useMemo(() => {
    if (!example || !outputData) return false
    try {
      return JSON.stringify(outputData) !== JSON.stringify(example.output_data || {})
    } catch (e) {
      return false
    }
  }, [outputData, example])

  // Keyboard navigation handler
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't handle keyboard shortcuts when user is typing in input fields
      // Exception: Allow Shift+Enter and Shift+G for generate from anywhere
      const target = e.target
      const isInputField = target.tagName === 'INPUT' || 
                          target.tagName === 'TEXTAREA' || 
                          target.isContentEditable ||
                          (target.tagName === 'SELECT' && e.key !== 'ArrowUp' && e.key !== 'ArrowDown')
      
      const isGenerateKey = (e.key === 'g' || e.key === 'G' || e.key === 'Enter') && e.shiftKey
      
      // Block all shortcuts in input fields except generate (Shift+G/Shift+Enter) and arrow keys for status
      if (isInputField && !isGenerateKey && e.key !== 'ArrowUp' && e.key !== 'ArrowDown') {
        return
      }

      // Left arrow: previous example
      if (e.key === 'ArrowLeft' && hasPrevious && previousExample) {
        e.preventDefault()
        navigate(`/tasks/${taskId}/examples/${previousExample.id}`, {
          state: { filters: filterParams },
          replace: true,
        })
      }
      // Right arrow: next example
      else if (e.key === 'ArrowRight' && hasNext && nextExample) {
        e.preventDefault()
        navigate(`/tasks/${taskId}/examples/${nextExample.id}`, {
          state: { filters: filterParams },
          replace: true,
        })
      }
      // Up arrow: previous status
      else if (e.key === 'ArrowUp') {
        e.preventDefault()
        const newIndex = selectedStatusIndex > 0 ? selectedStatusIndex - 1 : statusOptions.length - 1
        setSelectedStatusIndex(newIndex)
        updateStatusMutation.mutate(statusOptions[newIndex] || null)
      }
      // Down arrow: next status
      else if (e.key === 'ArrowDown') {
        e.preventDefault()
        const newIndex = selectedStatusIndex < statusOptions.length - 1 ? selectedStatusIndex + 1 : 0
        setSelectedStatusIndex(newIndex)
        updateStatusMutation.mutate(statusOptions[newIndex] || null)
      }
      // Enter: save changes
      else if (e.key === 'Enter' && !e.shiftKey && hasChanges && !updateMutation.isLoading) {
        e.preventDefault()
        handleSave()
      }
      // Shift+G or Shift+Enter: generate output
      else if ((e.key === 'g' || e.key === 'G' || e.key === 'Enter') && e.shiftKey && !generateOutputMutation.isLoading) {
        e.preventDefault()
        handleGenerate()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [hasPrevious, hasNext, previousExample, nextExample, taskId, filterParams, selectedStatusIndex, hasChanges, updateMutation, updateStatusMutation, generateOutputMutation, handleSave, handleGenerate, navigate])

  // Initialize outputData from example when it loads
  useEffect(() => {
    if (example?.output_data) {
      if (outputData === null) {
        setOutputData(example.output_data)
      }
    } else if (example && outputData === null) {
      // Initialize with empty object if no output_data exists
      setOutputData({})
    }
  }, [example, outputData])

  if (taskLoading || exampleLoading) {
    return <div className="text-center py-12">Loading example...</div>
  }

  if (!task || !example) {
    return <div className="text-center py-12">Example not found</div>
  }

  return (
    <div className="fixed inset-0 top-16 left-0 right-0 bottom-0 flex flex-col bg-gray-50 overflow-hidden">
      <div className="flex-shrink-0 px-2 py-1 border-b border-gray-200 bg-white">
        <div className="flex items-center justify-between">
          <button
            onClick={() => navigate(`/tasks/${taskId}`)}
            className="text-xs text-blue-600 hover:text-blue-900 inline-flex items-center px-2 py-1"
          >
            ← Back
          </button>
          <div className="flex items-center space-x-1">
            <button
              onClick={() => navigate(`/tasks/${taskId}/examples/${exampleId}/view`, { state: { filters: filterParams } })}
              className="px-2 py-1 text-xs border border-gray-300 rounded text-gray-700 hover:bg-gray-50 inline-flex items-center"
              title="View Only"
            >
              <svg
                className="w-3 h-3 mr-1"
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
            </button>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-2 py-1 text-xs border border-gray-300 rounded text-gray-700 hover:bg-gray-50 inline-flex items-center"
              title="Toggle Filters"
            >
              <svg
                className="w-3 h-3 mr-1"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                />
              </svg>
              Filters
            </button>
            <button
              onClick={() => {
                if (previousExample) {
                  navigate(`/tasks/${taskId}/examples/${previousExample.id}`, {
                    state: { filters: filterParams },
                    replace: true,
                  })
                }
              }}
              disabled={!hasPrevious}
              className="px-2 py-1 text-xs border border-gray-300 rounded text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center"
            >
              <svg
                className="w-3 h-3 mr-1"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 19l-7-7 7-7"
                />
              </svg>
              Prev
            </button>
            <span className="text-xs text-gray-500 px-1">
              {currentIndex >= 0 && examplesList
                ? `${currentIndex + 1}/${examplesList.length}`
                : ''}
            </span>
            <button
              onClick={() => {
                if (nextExample) {
                  navigate(`/tasks/${taskId}/examples/${nextExample.id}`, {
                    state: { filters: filterParams },
                    replace: true,
                  })
                }
              }}
              disabled={!hasNext}
              className="px-2 py-1 text-xs border border-gray-300 rounded text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center"
            >
              Next
              <svg
                className="w-3 h-3 ml-1"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </button>
          </div>
        </div>
        
        {showFilters && (
          <div className="mt-2 pb-1">
            <ExampleFilters
              filters={filterParams}
              onFilterChange={handleFilterChange}
              onClearFilters={handleClearFilters}
            />
          </div>
        )}
      </div>

      <div ref={containerRef} className="flex gap-0 flex-1 overflow-hidden" style={{ width: '100%' }}>
        {/* Input Section */}
        <div 
          className="bg-white border-r border-gray-200 overflow-hidden flex flex-col"
          style={{ width: `${splitPosition}%`, minWidth: 0, maxWidth: `${splitPosition}%` }}
        >
          <div className="flex-shrink-0 px-3 py-2 border-b border-gray-200 flex items-center justify-between bg-gray-50" style={{ minHeight: '40px', height: '40px' }}>
            <h2 className="text-sm font-medium text-gray-900">Input</h2>
          </div>
          <div className="flex-1 overflow-y-auto overflow-x-hidden p-3" style={{ width: '100%', minWidth: 0 }}>
            <div style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}>
              <ContentViewer inputData={example.input_data} inputSchema={task.input_schema} />
            </div>
          </div>
        </div>

        {/* Resizer */}
        <div
          className="w-0.5 bg-gray-300 hover:bg-blue-500 cursor-col-resize flex-shrink-0 relative group"
          onMouseDown={(e) => {
            e.preventDefault()
            setIsResizing(true)
          }}
        >
          <div className="absolute inset-y-0 left-1/2 transform -translate-x-1/2 w-8 -ml-4" />
        </div>

        {/* Output/Annotation Section */}
        <div 
          className="bg-white overflow-hidden flex flex-col"
          style={{ width: `${100 - splitPosition}%`, minWidth: 0, maxWidth: `${100 - splitPosition}%` }}
        >
          <div className="flex-shrink-0 px-3 py-2 border-b border-gray-200 flex items-center justify-between bg-gray-50" style={{ minHeight: '40px', height: '40px' }}>
            <h2 className="text-sm font-medium text-gray-900">Output / Annotation</h2>
            <div className="flex items-center space-x-2">
              {hasChanges && (
                <span className="text-xs text-orange-600 bg-orange-50 px-1.5 py-0.5 rounded">
                  Unsaved
                </span>
              )}
              <select
                value={example.status || ''}
                onChange={(e) => updateStatusMutation.mutate(e.target.value || null)}
                className="px-2 py-1 text-xs border border-gray-300 rounded"
              >
                <option value="">No Status</option>
                <option value="approved">Approved</option>
                <option value="rejected">Rejected</option>
                <option value="pending">Pending</option>
                <option value="reviewed">Reviewed</option>
              </select>
              <button
                onClick={handleGenerate}
                disabled={generateOutputMutation.isLoading}
                className="px-2 py-1 text-xs border border-gray-300 rounded bg-purple-600 text-white hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center justify-center gap-1 relative"
                title={generateOutputMutation.isLoading ? "Generation in progress..." : (task?.default_model ? "Generate output using dspydantic with most recent prompts (Shift+G)" : "Configure model in task settings first")}
              >
                {generateOutputMutation.isLoading ? (
                  <>
                    <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span className="text-[10px]">Generating...</span>
                  </>
                ) : (
                  <>
                    <span>✨</span>
                    <span className="text-[10px] opacity-75">⇧G</span>
                  </>
                )}
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto overflow-x-hidden p-3 relative" style={{ width: '100%', minWidth: 0 }}>
            {generateOutputMutation.isLoading && (
              <div className="absolute inset-0 bg-white bg-opacity-90 z-20 flex items-center justify-center backdrop-blur-sm">
                <div className="text-center bg-white rounded-lg shadow-lg p-6 border border-purple-200">
                  <svg className="animate-spin h-10 w-10 text-purple-600 mx-auto mb-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <p className="text-base font-semibold text-gray-900">Generating output...</p>
                  <p className="text-sm text-gray-600 mt-2">Please wait, this may take a moment</p>
                  <p className="text-xs text-gray-500 mt-1">Do not start another generation until this completes</p>
                </div>
              </div>
            )}
            <div className={`border border-gray-300 rounded p-3 bg-gray-50 ${generateOutputMutation.isLoading ? 'pointer-events-none opacity-50' : ''}`} style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}>
              <SchemaForm
                schema={task.pydantic_schema}
                data={outputData || {}}
                onChange={setOutputData}
              />
            </div>
          </div>
          <div className="flex-shrink-0 px-3 py-2 border-t border-gray-200 bg-gray-50 flex justify-end space-x-2">
            <button
              onClick={() => {
                setOutputData(example.output_data)
              }}
              disabled={!hasChanges || generateOutputMutation.isLoading}
              className="px-3 py-1 text-xs border border-gray-300 rounded text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Reset
            </button>
            <button
              onClick={handleSave}
              disabled={!hasChanges || updateMutation.isLoading || generateOutputMutation.isLoading}
              className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-1.5"
            >
              {updateMutation.isLoading ? 'Saving...' : (
                <>
                  <span>Save</span>
                  <span className="text-[10px] opacity-75">↵</span>
                </>
              )}
            </button>
          </div>
          {updateMutation.isSuccess && (
            <div className="px-3 py-1 text-xs text-green-600 bg-green-50">✓ Saved successfully</div>
          )}
          {updateMutation.isError && (
            <div className="px-3 py-1 text-xs text-red-600 bg-red-50">
              Error: {updateMutation.error?.response?.data?.detail || 'Failed to save'}
            </div>
          )}
        </div>
      </div>

      {showConfigureModelModal && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full shadow-xl">
            <div className="flex items-center space-x-2 mb-4">
              <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <h3 className="text-lg font-medium text-gray-900">Model Not Configured</h3>
            </div>
            <p className="text-sm text-gray-600 mb-4">
              Please configure a default model in the task settings before generating output.
            </p>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => setShowConfigureModelModal(false)}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowConfigureModelModal(false)
                  navigate(`/tasks/${taskId}`, { state: { activeTab: 'settings' } })
                }}
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500"
              >
                Go to Task Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
