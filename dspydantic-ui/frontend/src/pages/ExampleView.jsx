import { useState, useEffect } from 'react'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { examplesApi, tasksApi } from '../services/api'
import ContentViewer from '../components/ContentViewer'
import OutputViewer from '../components/OutputViewer'
import ExampleFilters from '../components/ExampleFilters'

export default function ExampleView() {
  const { taskId, exampleId } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [showFilters, setShowFilters] = useState(false)
  
  // Get filter params from location state or initialize empty
  const [filterParams, setFilterParams] = useState(location.state?.filters || {
    input_complete: null,
    output_complete: null,
    status: null,
    created_after: '',
    created_before: '',
  })

  const { data: task, isLoading: taskLoading } = useQuery({
    queryKey: ['tasks', taskId],
    queryFn: async () => {
      const response = await tasksApi.get(taskId)
      return response.data
    },
  })

  const { data: example, isLoading: exampleLoading } = useQuery({
    queryKey: ['examples', exampleId],
    queryFn: async () => {
      const response = await examplesApi.get(exampleId)
      return response.data
    },
  })

  // Get list of examples for navigation
  const { data: examplesList } = useQuery({
    queryKey: ['examples', taskId, filterParams],
    queryFn: async () => {
      const params = {
        limit: 1000,
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
  }

  // Navigate to first example when filters change and current example is not in filtered list
  useEffect(() => {
    if (examplesList && examplesList.length > 0 && currentIndex === -1 && exampleId) {
      const firstExampleId = examplesList[0].id
      if (parseInt(exampleId) !== firstExampleId) {
        navigate(`/tasks/${taskId}/examples/${firstExampleId}/view`, {
          state: { filters: filterParams },
          replace: true,
        })
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [examplesList, currentIndex])

  if (taskLoading || exampleLoading) {
    return <div className="text-center py-12">Loading example...</div>
  }

  if (!task || !example) {
    return <div className="text-center py-12">Example not found</div>
  }

  return (
    <div className="w-full px-4 sm:px-6 lg:px-8">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <button
            onClick={() => navigate(`/tasks/${taskId}`)}
            className="text-sm text-blue-600 hover:text-blue-900 inline-flex items-center"
          >
            ← Back to Task
          </button>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => navigate(`/tasks/${taskId}/examples/${exampleId}`, { state: { filters: filterParams } })}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 inline-flex items-center"
              title="Edit Mode"
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
              Edit
            </button>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 inline-flex items-center"
              title="Toggle Filters"
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
                  d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                />
              </svg>
              Filters
            </button>
            <button
              onClick={() => {
                if (previousExample) {
                  navigate(`/tasks/${taskId}/examples/${previousExample.id}/view`, {
                    state: { filters: filterParams },
                    replace: true,
                  })
                }
              }}
              disabled={!hasPrevious}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center"
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
                  d="M15 19l-7-7 7-7"
                />
              </svg>
              Previous
            </button>
            <span className="text-sm text-gray-500">
              {currentIndex >= 0 && examplesList
                ? `${currentIndex + 1} of ${examplesList.length}`
                : ''}
            </span>
            <button
              onClick={() => {
                if (nextExample) {
                  navigate(`/tasks/${taskId}/examples/${nextExample.id}/view`, {
                    state: { filters: filterParams },
                    replace: true,
                  })
                }
              }}
              disabled={!hasNext}
              className="px-3 py-1 text-sm border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center"
            >
              Next
              <svg
                className="w-4 h-4 ml-1"
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
          <div className="mb-4">
            <ExampleFilters
              filters={filterParams}
              onFilterChange={handleFilterChange}
              onClearFilters={handleClearFilters}
            />
          </div>
        )}
        
        <h1 className="text-2xl font-semibold text-gray-900">Example #{example.id} (View Only)</h1>
        <p className="mt-1 text-sm text-gray-500">
          Created: {new Date(example.created_at).toLocaleString()}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Section */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="mb-4">
            <h2 className="text-lg font-medium text-gray-900">Input</h2>
            <p className="text-sm text-gray-500">View the input content (PDF, image, or text)</p>
          </div>
          <div className="max-h-[calc(100vh-300px)] overflow-y-auto">
            <ContentViewer inputData={example.input_data} inputSchema={task.input_schema} />
          </div>
        </div>

        {/* Output Section */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div className="mb-4">
            <h2 className="text-lg font-medium text-gray-900">Output</h2>
            <p className="text-sm text-gray-500">View the output data</p>
          </div>
          <div className="max-h-[calc(100vh-300px)] overflow-y-auto">
            <OutputViewer outputData={example.output_data} outputSchema={task.pydantic_schema} />
          </div>
        </div>
      </div>
    </div>
  )
}
