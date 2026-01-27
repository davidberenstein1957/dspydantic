import { useState } from 'react'

const INPUT_FIELD_TYPES = [
  { value: 'string', label: 'String' },
  { value: 'pdf', label: 'PDF (URL or base64)' },
  { value: 'image', label: 'Image (URL or base64)' },
]

const OUTPUT_FIELD_TYPES = [
  { value: 'string', label: 'String' },
  { value: 'integer', label: 'Integer' },
  { value: 'number', label: 'Number' },
  { value: 'boolean', label: 'Boolean' },
  { value: 'array', label: 'Array' },
  { value: 'object', label: 'Object' },
]

const ARRAY_ITEM_TYPES = [
  { value: 'string', label: 'String' },
  { value: 'integer', label: 'Integer' },
  { value: 'number', label: 'Number' },
  { value: 'boolean', label: 'Boolean' },
  { value: 'object', label: 'Object' },
]

function FieldEditor({ field, onChange, onDelete, level = 0, isInputSchema = false, allowDescriptionEdit = true }) {
  const [isExpanded, setIsExpanded] = useState(true)
  const fieldTypes = isInputSchema ? INPUT_FIELD_TYPES : OUTPUT_FIELD_TYPES

  const handleTypeChange = (newType) => {
    let newField = { ...field, type: newType }
    
    // Reset type-specific properties
    if (newType === 'array') {
      // For input schemas, arrays are not allowed (only string, pdf, image)
      if (isInputSchema) {
        // Reset to string if trying to set array in input schema
        newField.type = 'string'
      } else {
        newField.items = { type: 'string' }
      }
    } else if (newType === 'object') {
      // For input schemas, objects are not allowed
      if (isInputSchema) {
        newField.type = 'string'
      } else {
        newField.properties = {}
      }
    } else {
      delete newField.items
      delete newField.properties
    }
    
    onChange(newField)
  }

  const handleNestedFieldChange = (fieldName, nestedField) => {
    const newProperties = { ...field.properties }
    if (nestedField === null) {
      delete newProperties[fieldName]
    } else {
      newProperties[fieldName] = nestedField
    }
    onChange({ ...field, properties: newProperties })
  }

  const addNestedField = () => {
    const newProperties = { ...field.properties || {} }
    const newFieldName = `field_${Object.keys(newProperties).length + 1}`
    newProperties[newFieldName] = { type: 'string', description: '', required: true }
    onChange({ ...field, properties: newProperties })
  }

  return (
    <div className={`border rounded-lg p-4 mb-3 ${level > 0 ? 'bg-gray-50' : 'bg-white'}`} style={{ marginLeft: `${level * 20}px` }}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Field Name</label>
            <input
              type="text"
              value={field.name || ''}
              onChange={(e) => onChange({ ...field, name: e.target.value })}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              placeholder="field_name"
            />
          </div>
          
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Type</label>
            <select
              value={field.type || 'string'}
              onChange={(e) => handleTypeChange(e.target.value)}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              {fieldTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          type="button"
          onClick={onDelete}
          className="ml-3 text-red-600 hover:text-red-900 text-sm"
        >
          ✕
        </button>
      </div>

      {allowDescriptionEdit && (
        <div className="mb-3">
          <label className="block text-xs font-medium text-gray-700 mb-1">Description</label>
          <textarea
            value={field.description || ''}
            onChange={(e) => onChange({ ...field, description: e.target.value })}
            rows={2}
            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            placeholder="Describe what this field represents..."
          />
        </div>
      )}

      {field.type === 'string' && (
        <div className="mb-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={field.render_markdown !== false}
              onChange={(e) => onChange({ ...field, render_markdown: e.target.checked })}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-xs text-gray-700">Enable markdown rendering (default: yes)</span>
          </label>
        </div>
      )}

      {!isInputSchema && field.type === 'string' && (
        <div className="mb-3">
          <label className="block text-xs font-medium text-gray-700 mb-1">
            Allowed Values (Enum)
            <span className="text-gray-500 font-normal ml-1">- One per line, leave empty for any string</span>
          </label>
          <textarea
            value={field._enumText !== undefined ? field._enumText : (field.enum ? field.enum.join('\n') : '')}
            onChange={(e) => {
              // Store raw text to allow newlines during editing
              onChange({ ...field, _enumText: e.target.value })
            }}
            onBlur={(e) => {
              // Parse enum values when user leaves the field
              const values = e.target.value.split('\n').map(v => v.trim()).filter(v => v !== '')
              const updatedField = { ...field }
              if (values.length > 0) {
                updatedField.enum = values
              } else {
                delete updatedField.enum
              }
              delete updatedField._enumText
              onChange(updatedField)
            }}
            rows={3}
            className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 font-mono"
            placeholder="positive&#10;negative&#10;neutral"
          />
        </div>
      )}

      {!isInputSchema && (field.type === 'integer' || field.type === 'number') && (
        <div className="mb-3 grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Minimum</label>
            <input
              type="number"
              value={field.minimum !== undefined ? field.minimum : ''}
              onChange={(e) => onChange({ ...field, minimum: e.target.value === '' ? undefined : Number(e.target.value) })}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              placeholder="No minimum"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Maximum</label>
            <input
              type="number"
              value={field.maximum !== undefined ? field.maximum : ''}
              onChange={(e) => onChange({ ...field, maximum: e.target.value === '' ? undefined : Number(e.target.value) })}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              placeholder="No maximum"
            />
          </div>
        </div>
      )}

      {field.type === 'array' && (
        <div className="mb-3 border-l-2 border-blue-200 pl-3">
          <label className="block text-xs font-medium text-gray-700 mb-2">Array Item Configuration</label>
          {!isInputSchema ? (
            <FieldEditor
              field={{ ...(field.items || { type: 'string' }), name: 'item' }}
              onChange={(updated) => {
                const { name, ...itemData } = updated
                onChange({ ...field, items: itemData })
              }}
              onDelete={() => {}}
              level={level + 1}
              isInputSchema={isInputSchema}
              allowDescriptionEdit={allowDescriptionEdit}
            />
          ) : (
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">Array Item Type</label>
              <select
                value={field.items?.type || 'string'}
                onChange={(e) => onChange({ ...field, items: { type: e.target.value } })}
                className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              >
                {INPUT_FIELD_TYPES.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      {field.type === 'object' && (
        <div className="mt-3">
          <div className="flex items-center justify-between mb-2">
            <label className="block text-xs font-medium text-gray-700">Nested Fields</label>
            <button
              type="button"
              onClick={addNestedField}
              className="text-xs text-blue-600 hover:text-blue-900"
            >
              + Add Field
            </button>
          </div>
          {field.properties && Object.entries(field.properties).map(([nestedName, nestedField]) => (
            <FieldEditor
              key={nestedName}
              field={{ ...nestedField, name: nestedName }}
              onChange={(updated) => {
                const newProperties = { ...field.properties }
                // Remove name from the field data before storing
                const { name, ...fieldData } = updated
                if (name && name !== nestedName) {
                  // Name changed, update the key
                  delete newProperties[nestedName]
                  newProperties[name] = fieldData
                } else {
                  newProperties[nestedName] = fieldData
                }
                onChange({ ...field, properties: newProperties })
              }}
              onDelete={() => handleNestedFieldChange(nestedName, null)}
              level={level + 1}
              isInputSchema={isInputSchema}
              allowDescriptionEdit={allowDescriptionEdit}
            />
          ))}
        </div>
      )}

      <div className="mt-3">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={field.required !== false}
            onChange={(e) => onChange({ ...field, required: e.target.checked })}
            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <span className="ml-2 text-xs text-gray-700">Required (default: yes)</span>
        </label>
      </div>
    </div>
  )
}

export default function SchemaBuilder({ schema, onChange, isInputSchema = false, allowDescriptionEdit = true }) {
  const parseFieldFromSchema = (name, def) => {
    const field = {
      name,
      type: def.type || 'string',
      description: def.description || '',
      required: def.required !== false,
    }

    // Parse render_markdown setting (defaults to true if not specified)
    if (def.type === 'string' && def.render_markdown !== undefined) {
      field.render_markdown = def.render_markdown
    }

    // Parse enum for strings
    if (def.type === 'string' && def.enum) {
      field.enum = def.enum
    }

    // Parse min/max for integers and numbers
    if ((def.type === 'integer' || def.type === 'number') && def.minimum !== undefined) {
      field.minimum = def.minimum
    }
    if ((def.type === 'integer' || def.type === 'number') && def.maximum !== undefined) {
      field.maximum = def.maximum
    }

    // Parse array items
    if (def.type === 'array' && def.items) {
      field.items = parseFieldFromSchema('', def.items)
    }

    // Parse nested object properties
    if (def.type === 'object' && def.properties) {
      field.properties = {}
      Object.entries(def.properties).forEach(([nestedName, nestedDef]) => {
        field.properties[nestedName] = parseFieldFromSchema(nestedName, nestedDef)
      })
    }

    return field
  }

  const [fields, setFields] = useState(() => {
    if (!schema || !schema.properties) return []
    return Object.entries(schema.properties).map(([name, def]) => parseFieldFromSchema(name, def))
  })

  const [requiredFields, setRequiredFields] = useState(schema?.required || [])
  const [currentSchema, setCurrentSchema] = useState(schema || { properties: {}, required: [] })

  const handleFieldChange = (index, updatedField) => {
    const newFields = [...fields]
    newFields[index] = updatedField
    setFields(newFields)
    updateSchema(newFields, requiredFields)
  }

  const handleFieldDelete = (index) => {
    const field = fields[index]
    const newFields = fields.filter((_, i) => i !== index)
    setFields(newFields)
    setRequiredFields(requiredFields.filter((name) => name !== field.name))
    updateSchema(newFields, requiredFields.filter((name) => name !== field.name))
  }

  const addField = () => {
    const newField = { name: '', type: 'string', description: '', required: true }
    setFields([...fields, newField])
    updateSchema([...fields, newField], requiredFields)
  }

  const updateSchema = (currentFields, currentRequired) => {
    const properties = {}
    const required = []

    const buildFieldSchema = (field) => {
      const fieldSchema = {
        type: field.type,
      }

      if (field.description) {
        fieldSchema.description = field.description
      }

      // Add render_markdown setting for strings (only save if false, default is true)
      if (field.type === 'string' && field.render_markdown === false) {
        fieldSchema.render_markdown = false
      }

      // Add enum constraint for strings
      // Parse _enumText if present (user is editing), otherwise use parsed enum
      let enumValues = field.enum
      if (field.type === 'string' && field._enumText !== undefined) {
        // Parse the raw text into enum values
        enumValues = field._enumText.split('\n').map(v => v.trim()).filter(v => v !== '')
      }
      if (field.type === 'string' && enumValues && enumValues.length > 0) {
        fieldSchema.enum = enumValues
      }

      // Add min/max constraints for integers and numbers
      if (field.type === 'integer' || field.type === 'number') {
        if (field.minimum !== undefined) {
          fieldSchema.minimum = field.minimum
        }
        if (field.maximum !== undefined) {
          fieldSchema.maximum = field.maximum
        }
      }

      if (field.type === 'array' && field.items) {
        fieldSchema.items = buildFieldSchema(field.items)
      }

      if (field.type === 'object' && field.properties) {
        const nestedProperties = {}
        Object.entries(field.properties).forEach(([nestedName, nestedField]) => {
          // Remove name from nestedField before building schema (name is stored as the key)
          const { name, ...fieldData } = nestedField
          nestedProperties[nestedName] = buildFieldSchema(fieldData)
        })
        fieldSchema.properties = nestedProperties
      }

      return fieldSchema
    }

    currentFields.forEach((field) => {
      if (field.name) {
        properties[field.name] = buildFieldSchema(field)
        // Include in required if not explicitly set to false (default is required)
        if (field.required !== false) {
          required.push(field.name)
        }
      }
    })

    const newSchema = { properties, required }
    setCurrentSchema(newSchema)
    onChange(newSchema)
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-900">Schema Fields</h3>
        <button
          type="button"
          onClick={addField}
          className="text-sm text-blue-600 hover:text-blue-900 font-medium"
        >
          + Add Field
        </button>
      </div>

      {fields.length === 0 ? (
        <div className="text-center py-8 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          <p className="text-sm text-gray-500 mb-2">No fields defined</p>
          <button
            type="button"
            onClick={addField}
            className="text-sm text-blue-600 hover:text-blue-900"
          >
            Add your first field
          </button>
        </div>
      ) : (
        <div>
          {fields.map((field, index) => (
            <FieldEditor
              key={index}
              field={field}
              onChange={(updated) => handleFieldChange(index, updated)}
              onDelete={() => handleFieldDelete(index)}
              isInputSchema={isInputSchema}
              allowDescriptionEdit={allowDescriptionEdit}
            />
          ))}
        </div>
      )}

      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <label className="block text-xs font-medium text-gray-700 mb-2">JSON Preview</label>
        <pre className="text-xs text-gray-600 overflow-auto max-h-40">
          {JSON.stringify(currentSchema, null, 2)}
        </pre>
      </div>
    </div>
  )
}
