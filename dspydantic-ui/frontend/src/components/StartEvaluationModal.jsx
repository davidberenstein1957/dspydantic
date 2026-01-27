import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { evaluationApi, promptsApi } from '../services/api'

export default function StartEvaluationModal({ taskId, task, onClose, onSuccess, initialConfig }) {
  const [metric, setMetric] = useState(initialConfig?.metric || 'exact')
  const [modelId, setModelId] = useState(initialConfig?.model_id || task?.default_model || '')
  const [promptVersionId, setPromptVersionId] = useState(initialConfig?.prompt_version_id || null)
  const [maxExamples, setMaxExamples] = useState(initialConfig?.max_examples || null)
  const queryClient = useQueryClient()

  const { data: promptVersions } = useQuery({
    queryKey: ['prompts', taskId],
    queryFn: async () => {
      const response = await promptsApi.list(taskId)
      return response.data
    },
  })

  const triggerMutation = useMutation({
    mutationFn: (config) => evaluationApi.trigger(taskId, config),
    onSuccess: () => {
      queryClient.invalidateQueries(['evaluation-runs', taskId])
      onSuccess?.()
      onClose()
    },
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    const config = {
      metric,
      max_examples: maxExamples || undefined,
    }
    if (modelId) {
      config.model_id = modelId
    }
    if (promptVersionId) {
      config.prompt_version_id = promptVersionId
    }
    triggerMutation.mutate(config)
  }

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <h3 className="text-lg font-medium text-gray-900 mb-4">{initialConfig ? 'Duplicate Evaluation' : 'Start Evaluation'}</h3>
        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model ID
              </label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder={task?.default_model || "gpt-4o"}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              />
              <p className="mt-1 text-xs text-gray-500">
                {task?.default_model 
                  ? `Leave empty to use task default: ${task.default_model}`
                  : "Leave empty to use default or env var"}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Metric (Evaluation Function)
              </label>
              <select
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="exact">Exact Match</option>
                <option value="levenshtein">Levenshtein Distance</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Prompt Version
              </label>
              <select
                value={promptVersionId || ''}
                onChange={(e) => setPromptVersionId(e.target.value ? parseInt(e.target.value) : null)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="">Latest/Current (Recommended)</option>
                {promptVersions?.map((version) => (
                  <option key={version.id} value={version.id}>
                    Version {version.version_number} {version.is_active ? '(Active)' : ''}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Examples
              </label>
              <input
                type="number"
                value={maxExamples || ''}
                onChange={(e) => setMaxExamples(e.target.value ? parseInt(e.target.value) : null)}
                min="1"
                step="1"
                placeholder="All"
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              />
              <p className="mt-1 text-xs text-gray-500">
                Maximum examples to evaluate (leave empty for all)
              </p>
            </div>
          </div>
          
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> Only examples with "approved" status will be used for evaluation.
            </p>
          </div>

          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={triggerMutation.isLoading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
            >
              {triggerMutation.isLoading ? 'Starting...' : 'Start Evaluation'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
