import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { evaluationApi, tasksApi, examplesApi } from '../services/api'
import ContentViewer from '../components/ContentViewer'
import OutputViewer from '../components/OutputViewer'
import JsonDiffViewer from '../components/JsonDiffViewer'

export default function EvaluationResultsView() {
  const { taskId, runId } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [selectedResultIndex, setSelectedResultIndex] = useState(0)

  const { data: task, isLoading: taskLoading } = useQuery({
    queryKey: ['tasks', taskId],
    queryFn: async () => {
      const response = await tasksApi.get(taskId)
      return response.data
    },
  })

  const { data: run, isLoading: runLoading } = useQuery({
    queryKey: ['evaluation-runs', runId],
    queryFn: async () => {
      const response = await evaluationApi.get(runId)
      return response.data
    },
  })

  const { data: results, isLoading: resultsLoading } = useQuery({
    queryKey: ['evaluation-results', runId],
    queryFn: async () => {
      const response = await evaluationApi.getResults(runId)
      return response.data
    },
    enabled: !!runId,
  })

  // Fetch example data for selected result
  const selectedResult = results?.[selectedResultIndex]
  const { data: example } = useQuery({
    queryKey: ['examples', selectedResult?.example_id],
    queryFn: async () => {
      const response = await examplesApi.get(selectedResult.example_id)
      return response.data
    },
    enabled: !!selectedResult?.example_id,
  })

  // Retry mutation
  const retryMutation = useMutation({
    mutationFn: async (resultId) => {
      return await evaluationApi.retryExample(runId, resultId)
    },
    onSuccess: () => {
      // Refetch results and run data
      queryClient.invalidateQueries({ queryKey: ['evaluation-results', runId] })
      queryClient.invalidateQueries({ queryKey: ['evaluation-runs', runId] })
    },
  })

  if (taskLoading || runLoading || resultsLoading) {
    return <div className="text-center py-12">Loading evaluation results...</div>
  }

  if (!task || !run || !results) {
    return <div className="text-center py-12">Evaluation results not found</div>
  }

  const hasPrevious = selectedResultIndex > 0
  const hasNext = selectedResultIndex < results.length - 1
  const currentResult = results[selectedResultIndex]
  const failedResults = results.filter(r => r.error_message)
  const failedIndices = results.map((r, idx) => r.error_message ? idx : null).filter(idx => idx !== null)
  
  const goToNextFailed = () => {
    const currentFailedIdx = failedIndices.findIndex(idx => idx > selectedResultIndex)
    if (currentFailedIdx >= 0) {
      setSelectedResultIndex(failedIndices[currentFailedIdx])
    } else if (failedIndices.length > 0) {
      setSelectedResultIndex(failedIndices[0])
    }
  }

  return (
    <div className="w-full px-4 sm:px-6 lg:px-8">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <button
            onClick={() => navigate(`/tasks/${taskId}`, { state: { activeTab: 'evaluation' } })}
            className="text-sm text-blue-600 hover:text-blue-900 inline-flex items-center"
          >
            ← Back to Evaluation
          </button>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => {
                if (hasPrevious) {
                  setSelectedResultIndex(selectedResultIndex - 1)
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
              {selectedResultIndex + 1} of {results.length}
            </span>
            <button
              onClick={() => {
                if (hasNext) {
                  setSelectedResultIndex(selectedResultIndex + 1)
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
        
        <h1 className="text-2xl font-semibold text-gray-900">
          Evaluation Run #{run.id} - Example #{currentResult?.example_id}
        </h1>
        {run.error_message && (
          <div className="mt-2 bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="text-sm font-medium text-red-800 mb-1">Run Error</h3>
            <p className="text-sm text-red-700">{run.error_message}</p>
          </div>
        )}
        {failedResults.length > 0 && (
          <div className="mt-2 bg-yellow-50 border border-yellow-200 rounded-lg p-3 flex items-center justify-between">
            <span className="text-sm text-yellow-800">
              {failedResults.length} of {results.length} entries failed
            </span>
            <button
              onClick={goToNextFailed}
              className="text-sm text-yellow-800 hover:text-yellow-900 font-medium underline"
            >
              Go to next failed entry
            </button>
          </div>
        )}
        <div className="mt-2 flex items-center gap-2 flex-wrap">
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              run.status === 'completed'
                ? 'bg-green-100 text-green-800'
                : run.status === 'failed'
                ? 'bg-red-100 text-red-800'
                : 'bg-gray-100 text-gray-800'
            }`}
          >
            {run.status}
          </span>
          {run.prompt_version_id && (
            <span className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded">
              Prompt Version: {run.prompt_version_id}
            </span>
          )}
          {currentResult && (
            <>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${
                  currentResult.score === 1.0
                    ? 'bg-green-100 text-green-800'
                    : currentResult.score >= 0.7
                    ? 'bg-yellow-100 text-yellow-800'
                    : 'bg-red-100 text-red-800'
                }`}
              >
                Score: {(currentResult.score * 100).toFixed(1)}%
              </span>
              {currentResult.error_message && (
                <button
                  onClick={() => {
                    if (currentResult.id) {
                      retryMutation.mutate(currentResult.id)
                    }
                  }}
                  disabled={retryMutation.isPending}
                  className="px-2 py-1 text-xs font-medium text-blue-600 bg-blue-50 rounded hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {retryMutation.isPending ? 'Retrying...' : 'Retry'}
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {currentResult && (
        <div className="space-y-6">
          {currentResult.error_message && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="text-sm font-medium text-red-800 mb-2">Error</h3>
                  <p className="text-sm text-red-700">{currentResult.error_message}</p>
                </div>
                <button
                  onClick={() => {
                    if (currentResult.id) {
                      retryMutation.mutate(currentResult.id)
                    }
                  }}
                  disabled={retryMutation.isPending}
                  className="ml-4 px-3 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {retryMutation.isPending ? 'Retrying...' : 'Retry'}
                </button>
              </div>
            </div>
          )}

          {example && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Input</h2>
              <div className="max-h-[400px] overflow-y-auto">
                <ContentViewer inputData={example.input_data} inputSchema={task.input_schema} />
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Expected Output</h2>
              <div className="max-h-[600px] overflow-y-auto">
                {currentResult.expected_output ? (
                  <OutputViewer outputData={currentResult.expected_output} outputSchema={task.pydantic_schema} />
                ) : (
                  <div className="text-gray-500">No expected output</div>
                )}
              </div>
            </div>

            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Extracted Output</h2>
              <div className="max-h-[600px] overflow-y-auto">
                {currentResult.extracted_output ? (
                  <OutputViewer outputData={currentResult.extracted_output} outputSchema={task.pydantic_schema} />
                ) : (
                  <div className="text-gray-500">No extracted output</div>
                )}
              </div>
            </div>
          </div>

          {currentResult.extracted_output && currentResult.expected_output && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Difference</h2>
              <JsonDiffViewer
                oldValue={currentResult.expected_output}
                newValue={currentResult.extracted_output}
              />
            </div>
          )}

          {currentResult.differences && Object.keys(currentResult.differences).length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">Detailed Differences</h2>
              <div className="space-y-3">
                {currentResult.differences.values_changed && Object.keys(currentResult.differences.values_changed).length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Values Changed</h3>
                    <div className="bg-gray-50 rounded p-3 space-y-2">
                      {Object.entries(currentResult.differences.values_changed).map(([path, change]) => (
                        <div key={path} className="text-sm">
                          <span className="font-mono text-gray-600">{path}:</span>
                          <div className="ml-4 mt-1">
                            <div className="text-red-600 line-through">{change.old}</div>
                            <div className="text-green-600">{change.new}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {currentResult.differences.dictionary_item_added && currentResult.differences.dictionary_item_added.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Items Added</h3>
                    <div className="bg-green-50 rounded p-3">
                      {currentResult.differences.dictionary_item_added.map((item) => (
                        <div key={item} className="text-sm font-mono text-green-800">{item}</div>
                      ))}
                    </div>
                  </div>
                )}
                {currentResult.differences.dictionary_item_removed && currentResult.differences.dictionary_item_removed.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Items Removed</h3>
                    <div className="bg-red-50 rounded p-3">
                      {currentResult.differences.dictionary_item_removed.map((item) => (
                        <div key={item} className="text-sm font-mono text-red-800">{item}</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
