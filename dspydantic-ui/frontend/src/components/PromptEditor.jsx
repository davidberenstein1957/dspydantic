import { useState, useMemo } from 'react'
import SchemaDescriptionEditor from './SchemaDescriptionEditor'

/**
 * Extract string field names from input schema (excluding pdf and image fields)
 */
function extractStringFields(inputSchema) {
  if (!inputSchema || !inputSchema.properties) return []
  
  const stringFields = []
  for (const [fieldName, fieldDef] of Object.entries(inputSchema.properties)) {
    if (fieldDef.type === 'string') {
      stringFields.push(fieldName)
    }
  }
  return stringFields
}

/**
 * Extract placeholders from jinja-like template (e.g., {field_name})
 */
function extractPlaceholders(template) {
  if (!template) return []
  const placeholderPattern = /\{([^}]+)\}/g
  const matches = []
  let match
  while ((match = placeholderPattern.exec(template)) !== null) {
    matches.push(match[1])
  }
  return [...new Set(matches)] // Return unique placeholders
}

export default function PromptEditor({ systemPrompt, instructionPrompt, schema, outputSchemaDescriptions, inputSchema, onChange, onSave, onCancel }) {
  const [localSystemPrompt, setLocalSystemPrompt] = useState(systemPrompt || '')
  const [localInstructionPrompt, setLocalInstructionPrompt] = useState(instructionPrompt || '')
  const [localDescriptions, setLocalDescriptions] = useState(outputSchemaDescriptions || {})

  // Extract available string fields from input schema
  const availableInputFields = useMemo(() => extractStringFields(inputSchema), [inputSchema])
  
  // Extract placeholders from instruction prompt
  const placeholders = useMemo(() => extractPlaceholders(localInstructionPrompt), [localInstructionPrompt])
  
  // Validate placeholders
  const invalidPlaceholders = useMemo(() => {
    return placeholders.filter(p => !availableInputFields.includes(p))
  }, [placeholders, availableInputFields])

  const handleSave = () => {
    const promptData = {
      system_prompt: localSystemPrompt,
      instruction_prompt: localInstructionPrompt,
      output_schema_descriptions: Object.keys(localDescriptions).length > 0 ? localDescriptions : null,
    }
    if (onChange) onChange(promptData)
    if (onSave) onSave(promptData)
  }

  const handleDescriptionsChange = (descriptions) => {
    setLocalDescriptions(descriptions)
    if (onChange) {
      onChange({
        system_prompt: localSystemPrompt,
        instruction_prompt: localInstructionPrompt,
        output_schema_descriptions: descriptions,
      })
    }
  }

  const handleInstructionPromptChange = (value) => {
    setLocalInstructionPrompt(value)
    if (onChange) {
      onChange({
        system_prompt: localSystemPrompt,
        instruction_prompt: value,
        output_schema_descriptions: localDescriptions,
      })
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          System Prompt
        </label>
        <textarea
          value={localSystemPrompt}
          onChange={(e) => setLocalSystemPrompt(e.target.value)}
          rows={6}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
          placeholder="Enter the system prompt that defines the assistant's behavior..."
        />
        <p className="mt-1 text-xs text-gray-500">
          This prompt sets the overall behavior and context for the assistant
        </p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Instruction Prompt Template
        </label>
        <textarea
          value={localInstructionPrompt}
          onChange={(e) => handleInstructionPromptChange(e.target.value)}
          rows={8}
          className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 font-mono text-sm ${
            invalidPlaceholders.length > 0 ? 'border-red-300' : 'border-gray-300'
          }`}
          placeholder="Enter the instruction prompt template. Use {variable_name} for placeholders..."
        />
        <p className="mt-1 text-xs text-gray-500">
          This is the instruction template that can include placeholders like {'{'}variable_name{'}'} for dynamic content
        </p>
        
        {invalidPlaceholders.length > 0 && (
          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded-md">
            <p className="text-xs font-medium text-red-800 mb-1">Invalid placeholders:</p>
            <ul className="text-xs text-red-700 list-disc list-inside">
              {invalidPlaceholders.map((placeholder) => (
                <li key={placeholder}>{'{'}{placeholder}{'}'} - not found in input schema</li>
              ))}
            </ul>
            <p className="text-xs text-red-600 mt-1">
              Only string fields from the input schema can be used as placeholders.
            </p>
          </div>
        )}
        
        {availableInputFields.length > 0 && (
          <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded-md">
            <p className="text-xs font-medium text-blue-800 mb-1">Available input fields (string only):</p>
            <div className="flex flex-wrap gap-2">
              {availableInputFields.map((fieldName) => (
                <code key={fieldName} className="text-xs bg-white px-2 py-1 rounded border border-blue-300 text-blue-700">
                  {'{'}{fieldName}{'}'}
                </code>
              ))}
            </div>
            <p className="text-xs text-blue-600 mt-1">
              You can use these placeholders in your instruction prompt template. PDF and image fields are not available.
            </p>
          </div>
        )}
        
        {availableInputFields.length === 0 && inputSchema && (
          <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded-md">
            <p className="text-xs text-yellow-800">
              No string fields found in input schema. Add string fields to the input schema to use them as placeholders.
            </p>
          </div>
        )}
      </div>

      {schema && (
        <div className="border-t border-gray-200 pt-4">
          <SchemaDescriptionEditor
            schema={schema}
            descriptions={localDescriptions}
            onChange={handleDescriptionsChange}
          />
        </div>
      )}

      <div className="flex justify-end space-x-3">
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            Cancel
          </button>
        )}
        <button
          type="button"
          onClick={handleSave}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-500"
        >
          Save Prompt
        </button>
      </div>
    </div>
  )
}
