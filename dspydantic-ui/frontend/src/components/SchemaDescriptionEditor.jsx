import { useState, useEffect } from 'react'

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

export default function SchemaDescriptionEditor({ schema, descriptions, onChange }) {
  const [localDescriptions, setLocalDescriptions] = useState(descriptions || {})

  useEffect(() => {
    if (descriptions) {
      setLocalDescriptions(descriptions)
    }
  }, [descriptions])

  const fieldPaths = extractFieldPaths(schema)

  const handleDescriptionChange = (fieldPath, value) => {
    const newDescriptions = { ...localDescriptions }
    if (value.trim()) {
      newDescriptions[fieldPath] = value
    } else {
      delete newDescriptions[fieldPath]
    }
    setLocalDescriptions(newDescriptions)
    onChange(newDescriptions)
  }

  if (!schema || !schema.properties || Object.keys(schema.properties).length === 0) {
    return (
      <div className="text-sm text-gray-500 italic">
        No output schema defined. Define the output schema in Task Settings first.
      </div>
    )
  }

  if (fieldPaths.length === 0) {
    return (
      <div className="text-sm text-gray-500 italic">
        No fields found in the output schema.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">
          Output Schema Field Descriptions
        </h4>
        <p className="text-xs text-gray-500 mb-4">
          Edit descriptions for output schema fields. These descriptions are used by the LLM to understand what each field should contain.
        </p>
      </div>
      <div className="space-y-3">
        {fieldPaths.map((fieldPath) => (
          <div key={fieldPath} className="border border-gray-200 rounded-md p-3 bg-gray-50">
            <label className="block text-xs font-medium text-gray-700 mb-1">
              {fieldPath}
            </label>
            <textarea
              value={localDescriptions[fieldPath] || ''}
              onChange={(e) => handleDescriptionChange(fieldPath, e.target.value)}
              rows={2}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter description for this field..."
            />
          </div>
        ))}
      </div>
    </div>
  )
}
