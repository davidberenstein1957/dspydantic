import { useState, useEffect } from 'react'
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { tasksApi, examplesApi } from '../services/api'
import SchemaBuilder from './SchemaBuilder'

export default function TaskSettingsTab({ task }) {
  const [isEditing, setIsEditing] = useState(false)
  const [name, setName] = useState(task.name || '')
  const [description, setDescription] = useState(task.description || '')
  const [defaultModel, setDefaultModel] = useState(task.default_model || '')
  const [inputSchema, setInputSchema] = useState(task.input_schema || { properties: {}, required: [] })
  const [outputSchema, setOutputSchema] = useState(task.pydantic_schema || { properties: {}, required: [] })
  const [schemaError, setSchemaError] = useState('')
  const [testResult, setTestResult] = useState(null)
  const [isTesting, setIsTesting] = useState(false)
  const queryClient = useQueryClient()
  const navigate = useNavigate()

  // Sync state when task prop changes (e.g., after update)
  useEffect(() => {
    if (!isEditing) {
      setName(task.name || '')
      setDescription(task.description || '')
      setDefaultModel(task.default_model || '')
      setInputSchema(task.input_schema || { properties: {}, required: [] })
      setOutputSchema(task.pydantic_schema || { properties: {}, required: [] })
    }
  }, [task, isEditing])

  // Get examples for testing
  const { data: examples } = useQuery({
    queryKey: ['examples', task.id],
    queryFn: async () => {
      const response = await examplesApi.list(task.id, { limit: 10 })
      return response.data
    },
    enabled: !!task.id,
  })

  const updateMutation = useMutation({
    mutationFn: (data) => tasksApi.update(task.id, data),
    onSuccess: async (_, variables) => {
      await queryClient.invalidateQueries(['tasks'])
      await queryClient.invalidateQueries(['tasks', task.id])
      queryClient.invalidateQueries(['examples', task.id])
      queryClient.invalidateQueries(['schema-validation', task.id])
      
      // Test the model if it was just configured
      if (variables.default_model && variables.default_model !== task.default_model) {
        // Wait a bit for queries to update
        setTimeout(() => {
          testModelConfiguration(variables.default_model)
        }, 500)
      }
      
      setIsEditing(false)
    },
  })

  const testModelConfiguration = async (modelId) => {
    if (!examples || examples.length === 0) {
      setTestResult({ success: false, message: 'No examples available to test. Create an example first.' })
      return
    }

    setIsTesting(true)
    setTestResult(null)

    try {
      // Find first example with input data
      const testExample = examples.find(ex => ex.input_data && Object.keys(ex.input_data).length > 0) || examples[0]
      
      if (!testExample) {
        setTestResult({ success: false, message: 'No examples with input data found.' })
        setIsTesting(false)
        return
      }

      // Try to generate output
      const response = await examplesApi.generateOutput(testExample.id, { 
        model_id: modelId, 
        api_key: undefined 
      })
      
      if (response.data && response.data.output_data) {
        setTestResult({ 
          success: true, 
          message: `Model test successful! Generated output for example #${testExample.id}.` 
        })
      } else {
        setTestResult({ success: false, message: 'Test completed but no output was generated.' })
      }
    } catch (error) {
      setTestResult({ 
        success: false, 
        message: error?.response?.data?.detail || error?.message || 'Failed to test model configuration.' 
      })
    } finally {
      setIsTesting(false)
    }
  }

  const deleteMutation = useMutation({
    mutationFn: () => tasksApi.delete(task.id),
    onSuccess: () => {
      queryClient.invalidateQueries(['tasks'])
      navigate('/tasks')
    },
  })

  const handleDelete = () => {
    if (window.confirm(`Are you sure you want to delete task "${task.name}"? This action cannot be undone.`)) {
      deleteMutation.mutate()
    }
  }

  const handleSave = (e) => {
    e.preventDefault()
    setSchemaError('')

    if (!name || !name.trim()) {
      setSchemaError('Task name is required')
      return
    }

    if (!outputSchema.properties || Object.keys(outputSchema.properties).length === 0) {
      setSchemaError('Output schema must have at least one field')
      return
    }

    updateMutation.mutate({
      name: name.trim(),
      description: description || null,
      default_model: defaultModel || null,
      input_schema: inputSchema.properties && Object.keys(inputSchema.properties).length > 0 ? inputSchema : null,
      pydantic_schema: outputSchema,
    })
  }

  const handleCancel = () => {
    setName(task.name || '')
    setDescription(task.description || '')
    setDefaultModel(task.default_model || '')
    setInputSchema(task.input_schema || { properties: {}, required: [] })
    setOutputSchema(task.pydantic_schema || { properties: {}, required: [] })
    setSchemaError('')
    setTestResult(null)
    setIsEditing(false)
  }

  if (isEditing) {
    return (
      <div className="space-y-6">
        <div className="sm:flex sm:items-center sm:justify-between mb-4">
          <div className="sm:flex-auto">
            <h2 className="text-lg font-medium text-gray-900">Task Settings</h2>
            <p className="mt-1 text-sm text-gray-500">
              Configure task name, description, schemas, and default model
            </p>
          </div>
          <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
            <button
              type="button"
              onClick={handleCancel}
              className="inline-flex items-center justify-center rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 bg-white shadow-sm hover:bg-gray-50"
            >
              Cancel
            </button>
          </div>
        </div>
        <div className="bg-white shadow rounded-lg p-6">

          <form onSubmit={handleSave}>
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Task Name *
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Default Model (for output generation)
                </label>
                <input
                  type="text"
                  value={defaultModel}
                  onChange={(e) => setDefaultModel(e.target.value)}
                  placeholder="e.g., gpt-4o"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Default model ID to use when generating outputs for examples without output data
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Input Schema (Optional)
                </label>
                <p className="mb-2 text-xs text-gray-500">
                  Define input fields that can be strings, PDFs (URL or base64), or images (URL or base64)
                </p>
                <div className="border border-gray-300 rounded-md p-4 bg-white">
                  <SchemaBuilder schema={inputSchema} onChange={setInputSchema} isInputSchema={true} />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Output Schema (Pydantic) *
                </label>
                {schemaError && (
                  <p className="mb-2 text-sm text-red-600">{schemaError}</p>
                )}
                <p className="mb-2 text-xs text-gray-500">
                  Define the output structure that examples must match (strings, numbers, booleans, arrays, objects only)
                </p>
                <div className="border border-gray-300 rounded-md p-4 bg-white">
                  <SchemaBuilder schema={outputSchema} onChange={setOutputSchema} isInputSchema={false} allowDescriptionEdit={false} />
                  <p className="mt-2 text-xs text-gray-500 italic">
                    Note: Field descriptions can only be edited from the Prompt Versions tab. Here you can only edit field names, types, and constraints.
                  </p>
                </div>
              </div>

              {testResult && (
                <div className={`p-3 rounded-md ${
                  testResult.success 
                    ? 'bg-green-50 border border-green-200' 
                    : 'bg-red-50 border border-red-200'
                }`}>
                  <div className={`text-sm ${
                    testResult.success ? 'text-green-800' : 'text-red-800'
                  }`}>
                    {testResult.success ? (
                      <div className="flex items-center space-x-2">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>{testResult.message}</span>
                      </div>
                    ) : (
                      <div className="flex items-center space-x-2">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span>{testResult.message}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {isTesting && (
                <div className="p-3 rounded-md bg-blue-50 border border-blue-200">
                  <div className="text-sm text-blue-800 flex items-center space-x-2">
                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Testing model configuration...</span>
                  </div>
                </div>
              )}
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={handleCancel}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={updateMutation.isLoading || isTesting}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
                >
                  {updateMutation.isLoading ? 'Saving...' : 'Save Changes'}
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="sm:flex sm:items-center sm:justify-between mb-4">
        <div className="sm:flex-auto">
          <h2 className="text-lg font-medium text-gray-900">Task Settings</h2>
          <p className="mt-1 text-sm text-gray-500">
            Configure task name, description, schemas, and default model
          </p>
        </div>
        <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none">
          <button
            type="button"
            onClick={() => {
              setTestResult(null)
              setIsEditing(true)
            }}
            className="inline-flex items-center justify-center rounded-md border border-transparent bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-500"
          >
            Edit
          </button>
        </div>
      </div>
      <div className="bg-white shadow rounded-lg p-6">

        <div className="space-y-6">
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Task Name</h4>
            <p className="text-sm text-gray-900">{task.name}</p>
          </div>

          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Description</h4>
            {task.description ? (
              <p className="text-sm text-gray-900">{task.description}</p>
            ) : (
              <p className="text-sm text-gray-500 italic">No description</p>
            )}
          </div>

          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Default Model</h4>
            {task.default_model ? (
              <p className="text-sm text-gray-900">{task.default_model}</p>
            ) : (
              <p className="text-sm text-gray-500 italic">No default model set</p>
            )}
          </div>

          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Input Schema</h4>
            {task.input_schema && task.input_schema.properties && Object.keys(task.input_schema.properties).length > 0 ? (
              <div className="bg-gray-50 rounded-md p-4">
                <pre className="text-xs text-gray-800 overflow-auto">
                  {JSON.stringify(task.input_schema, null, 2)}
                </pre>
              </div>
            ) : (
              <p className="text-sm text-gray-500 italic">No input schema defined</p>
            )}
          </div>

          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Output Schema (Pydantic)</h4>
            <div className="bg-gray-50 rounded-md p-4">
              <pre className="text-xs text-gray-800 overflow-auto">
                {JSON.stringify(task.pydantic_schema, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white shadow rounded-lg p-6 border-t border-red-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-red-900">Danger Zone</h3>
            <p className="mt-1 text-sm text-gray-600">
              Once you delete a task, there is no going back. Please be certain.
            </p>
          </div>
          <button
            type="button"
            onClick={handleDelete}
            disabled={deleteMutation.isLoading}
            className="px-4 py-2 bg-red-600 text-white rounded-md text-sm font-medium hover:bg-red-700 disabled:opacity-50"
          >
            {deleteMutation.isLoading ? 'Deleting...' : 'Delete Task'}
          </button>
        </div>
      </div>
    </div>
  )
}
