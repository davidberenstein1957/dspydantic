import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { tasksApi } from '../services/api'
import SchemaBuilder from './SchemaBuilder'

export default function CreateTaskModal({ onClose, onSuccess }) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [inputSchema, setInputSchema] = useState({ properties: {}, required: [] })
  const [outputSchema, setOutputSchema] = useState({ properties: {}, required: [] })
  const [systemPrompt, setSystemPrompt] = useState('')
  const [instructionPromptTemplate, setInstructionPromptTemplate] = useState('')
  const [schemaError, setSchemaError] = useState('')

  const createMutation = useMutation({
    mutationFn: (data) => tasksApi.create(data),
    onSuccess: () => {
      onSuccess()
    },
  })

  const handleSubmit = (e) => {
    e.preventDefault()
    setSchemaError('')

    if (!outputSchema.properties || Object.keys(outputSchema.properties).length === 0) {
      setSchemaError('Please add at least one field to the output schema')
      return
    }

    createMutation.mutate({
      name,
      description,
      input_schema: inputSchema.properties && Object.keys(inputSchema.properties).length > 0 ? inputSchema : null,
      pydantic_schema: outputSchema,
      system_prompt: systemPrompt || null,
      instruction_prompt_template: instructionPromptTemplate || null,
    })
  }

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-xl font-semibold mb-4">Create New Task</h2>

          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Name *
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="mb-4">
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

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                System Prompt
              </label>
              <textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                placeholder="Enter the system prompt that defines the assistant's behavior..."
              />
              <p className="mt-1 text-xs text-gray-500">
                This prompt sets the overall behavior and context for the assistant
              </p>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Instruction Prompt Template
              </label>
              <textarea
                value={instructionPromptTemplate}
                onChange={(e) => setInstructionPromptTemplate(e.target.value)}
                rows={6}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                placeholder="Enter the instruction prompt template. Use {variable_name} for placeholders..."
              />
              <p className="mt-1 text-xs text-gray-500">
                This is the instruction template that can include placeholders like {'{'}variable_name{'}'} for dynamic content
              </p>
            </div>

            <div className="mb-4">
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

            <div className="mb-4">
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
                <SchemaBuilder schema={outputSchema} onChange={setOutputSchema} isInputSchema={false} />
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
                disabled={createMutation.isLoading}
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
              >
                {createMutation.isLoading ? 'Creating...' : 'Create Task'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
