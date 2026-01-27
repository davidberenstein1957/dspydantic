import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import { examplesApi } from '../services/api'
import SchemaForm from './SchemaForm'

export default function ExampleModal({ taskId, task, example, onClose, onSuccess }) {
  const [inputData, setInputData] = useState(example?.input_data || {})
  const [outputData, setOutputData] = useState(example?.output_data || {})

  const createMutation = useMutation({
    mutationFn: (data) => examplesApi.create(taskId, data),
    onSuccess,
  })

  const updateMutation = useMutation({
    mutationFn: (data) => examplesApi.update(example.id, data),
    onSuccess,
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    const data = { input_data: inputData, output_data: outputData }
    if (example) {
      updateMutation.mutate(data)
    } else {
      createMutation.mutate(data)
    }
  }

  // Use task's input_schema if available, otherwise default to empty
  const inputSchema = task.input_schema || { properties: {}, required: [] }

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4">
            {example ? 'Edit Example' : 'Add Example'}
          </h2>

          <form onSubmit={handleSubmit}>
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Input Data
              </label>
              <div className="border border-gray-300 rounded-md p-4 bg-gray-50">
                <SchemaForm
                  schema={inputSchema}
                  data={inputData}
                  onChange={setInputData}
                />
              </div>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Output Data (must match task schema)
              </label>
              <div className="border border-gray-300 rounded-md p-4 bg-gray-50">
                <SchemaForm
                  schema={task.pydantic_schema}
                  data={outputData}
                  onChange={setOutputData}
                />
              </div>
            </div>

            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={createMutation.isLoading || updateMutation.isLoading}
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
              >
                {example
                  ? updateMutation.isLoading
                    ? 'Updating...'
                    : 'Update'
                  : createMutation.isLoading
                  ? 'Creating...'
                  : 'Create'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
