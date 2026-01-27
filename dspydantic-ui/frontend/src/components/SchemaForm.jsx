import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

export default function SchemaForm({ schema, data, onChange, prefix = '' }) {
  const [formData, setFormData] = useState(data || {})

  useEffect(() => {
    if (data) {
      setFormData(data)
    }
  }, [data])

  const handleChange = (fieldName, value) => {
    const newData = { ...formData, [fieldName]: value }
    setFormData(newData)
    onChange(newData)
  }

  const renderField = (fieldName, fieldDef) => {
    const fieldType = fieldDef.type || 'string'
    const isRequired = schema.required?.includes(fieldName) || false
    const fullFieldName = prefix ? `${prefix}.${fieldName}` : fieldName

    switch (fieldType) {
      case 'string':
        return (
          <div key={fieldName} className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {fieldName} {isRequired && <span className="text-red-500">*</span>}
            </label>
            <textarea
              value={formData[fieldName] || ''}
              onChange={(e) => handleChange(fieldName, e.target.value)}
              required={isRequired}
              rows={4}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
              style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
            />
            {formData[fieldName] && (
              <div className="mt-2 border border-gray-200 rounded-md p-3 bg-gray-50">
                <div className="text-xs text-gray-500 mb-1">Preview:</div>
                {fieldDef?.render_markdown !== false ? (
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown>{String(formData[fieldName])}</ReactMarkdown>
                  </div>
                ) : (
                  <div className="text-sm text-gray-800 whitespace-pre-wrap">{String(formData[fieldName])}</div>
                )}
              </div>
            )}
          </div>
        )

      case 'pdf':
      case 'image':
        const handleFileUpload = (e) => {
          const file = e.target.files[0]
          if (file) {
            const reader = new FileReader()
            reader.onload = (event) => {
              const dataUri = event.target.result
              handleChange(fieldName, dataUri)
            }
            reader.readAsDataURL(file)
          }
        }

        const isDataUri = formData[fieldName]?.startsWith('data:')
        const isUrl = formData[fieldName]?.startsWith('http://') || formData[fieldName]?.startsWith('https://')

        return (
          <div key={fieldName} className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {fieldName} ({fieldType.toUpperCase()}) {isRequired && <span className="text-red-500">*</span>}
            </label>
            <div className="space-y-2">
              <input
                type="text"
                value={formData[fieldName] || ''}
                onChange={(e) => handleChange(fieldName, e.target.value)}
                placeholder={`Enter ${fieldType} URL or upload file`}
                required={isRequired}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              />
              <div className="flex items-center space-x-2">
                <label className="flex items-center px-3 py-2 border border-gray-300 rounded-md cursor-pointer hover:bg-gray-50 text-sm">
                  <input
                    type="file"
                    accept={fieldType === 'pdf' ? 'application/pdf' : 'image/*'}
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <span>Upload {fieldType === 'pdf' ? 'PDF' : 'Image'}</span>
                </label>
                {isDataUri && (
                  <span className="text-xs text-green-600">✓ File uploaded (base64)</span>
                )}
                {isUrl && (
                  <span className="text-xs text-blue-600">✓ URL provided</span>
                )}
              </div>
              {isDataUri && fieldType === 'image' && (
                <div className="mt-2">
                  <img
                    src={formData[fieldName]}
                    alt="Preview"
                    className="max-w-xs max-h-48 border border-gray-300 rounded"
                  />
                </div>
              )}
            </div>
          </div>
        )

      case 'integer':
      case 'number':
        return (
          <div key={fieldName} className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {fieldName} {isRequired && <span className="text-red-500">*</span>}
            </label>
            <input
              type="number"
              value={formData[fieldName] || ''}
              onChange={(e) =>
                handleChange(
                  fieldName,
                  fieldType === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value)
                )
              }
              required={isRequired}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
            />
          </div>
        )

      case 'boolean':
        return (
          <div key={fieldName} className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData[fieldName] || false}
                onChange={(e) => handleChange(fieldName, e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-700">{fieldName}</span>
            </label>
          </div>
        )

      case 'array':
        return (
          <div key={fieldName} className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {fieldName} {isRequired && <span className="text-red-500">*</span>}
            </label>
            <div className="space-y-2">
              {(formData[fieldName] || []).map((item, idx) => (
                <div key={idx} className="flex items-center space-x-2">
                  <input
                    type="text"
                    value={item}
                    onChange={(e) => {
                      const newArray = [...(formData[fieldName] || [])]
                      newArray[idx] = e.target.value
                      handleChange(fieldName, newArray)
                    }}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm"
                    style={{ minWidth: 0, maxWidth: '100%', boxSizing: 'border-box' }}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      const newArray = [...(formData[fieldName] || [])]
                      newArray.splice(idx, 1)
                      handleChange(fieldName, newArray)
                    }}
                    className="text-red-600 hover:text-red-900"
                  >
                    Remove
                  </button>
                </div>
              ))}
              <button
                type="button"
                onClick={() => {
                  handleChange(fieldName, [...(formData[fieldName] || []), ''])
                }}
                className="text-sm text-blue-600 hover:text-blue-900"
              >
                + Add Item
              </button>
            </div>
          </div>
        )

      case 'object':
        return (
          <div key={fieldName} className="mb-4 border-l-2 border-gray-200 pl-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {fieldName} {isRequired && <span className="text-red-500">*</span>}
            </label>
            <SchemaForm
              schema={fieldDef}
              data={formData[fieldName] || {}}
              onChange={(nestedData) => handleChange(fieldName, nestedData)}
              prefix={fullFieldName}
            />
          </div>
        )

      default:
        return (
          <div key={fieldName} className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">{fieldName}</label>
            <input
              type="text"
              value={formData[fieldName] || ''}
              onChange={(e) => handleChange(fieldName, e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm"
              style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
            />
          </div>
        )
    }
  }

  if (!schema || !schema.properties) {
    return <div className="text-sm text-gray-500">No schema defined</div>
  }

  return (
    <div style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}>
      {Object.entries(schema.properties).map(([fieldName, fieldDef]) =>
        renderField(fieldName, fieldDef)
      )}
    </div>
  )
}
