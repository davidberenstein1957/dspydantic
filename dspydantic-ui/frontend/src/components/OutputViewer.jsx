import ReactMarkdown from 'react-markdown'

export default function OutputViewer({ outputData, outputSchema }) {
  if (!outputData || (typeof outputData !== 'object')) {
    return <div className="text-gray-500">No output data available</div>
  }

  const renderField = (fieldName, fieldValue, fieldDef) => {
    if (fieldValue === undefined || fieldValue === null) {
      return null
    }

    const fieldType = fieldDef?.type || (typeof fieldValue === 'string' ? 'string' : typeof fieldValue)

    switch (fieldType) {
      case 'string':
        const shouldRenderMarkdown = fieldDef?.render_markdown !== false
        return (
          <div key={fieldName} className="mb-4">
            <div className="text-sm font-medium text-gray-700 mb-2">{fieldName}</div>
            <div className="bg-gray-50 rounded-lg p-4 prose prose-sm max-w-none">
              {shouldRenderMarkdown ? (
                <ReactMarkdown>{String(fieldValue)}</ReactMarkdown>
              ) : (
                <div className="text-sm text-gray-800 whitespace-pre-wrap">{String(fieldValue)}</div>
              )}
            </div>
          </div>
        )

      case 'number':
      case 'integer':
        return (
          <div key={fieldName} className="mb-4">
            <div className="text-sm font-medium text-gray-700 mb-2">{fieldName}</div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-800">{String(fieldValue)}</div>
            </div>
          </div>
        )

      case 'boolean':
        return (
          <div key={fieldName} className="mb-4">
            <div className="text-sm font-medium text-gray-700 mb-2">{fieldName}</div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-800">{String(fieldValue)}</div>
            </div>
          </div>
        )

      case 'array':
        const arrayItemDef = fieldDef?.items
        const shouldRenderArrayMarkdown = arrayItemDef?.type === 'string' && arrayItemDef?.render_markdown !== false
        return (
          <div key={fieldName} className="mb-4">
            <div className="text-sm font-medium text-gray-700 mb-2">{fieldName}</div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="space-y-2">
                {Array.isArray(fieldValue) && fieldValue.map((item, idx) => (
                  <div key={idx} className="border-l-2 border-gray-300 pl-3">
                    {typeof item === 'string' ? (
                      shouldRenderArrayMarkdown ? (
                        <div className="prose prose-sm max-w-none">
                          <ReactMarkdown>{String(item)}</ReactMarkdown>
                        </div>
                      ) : (
                        <div className="text-sm text-gray-800 whitespace-pre-wrap">{String(item)}</div>
                      )
                    ) : (
                      <div className="text-sm text-gray-800">{JSON.stringify(item, null, 2)}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )

      case 'object':
        return (
          <div key={fieldName} className="mb-4 border-l-2 border-gray-200 pl-4">
            <div className="text-sm font-medium text-gray-700 mb-2">{fieldName}</div>
            <div className="space-y-2">
              {Object.entries(fieldValue || {}).map(([nestedName, nestedValue]) =>
                renderField(nestedName, nestedValue, fieldDef?.properties?.[nestedName])
              )}
            </div>
          </div>
        )

      default:
        return (
          <div key={fieldName} className="mb-4">
            <div className="text-sm font-medium text-gray-700 mb-2">{fieldName}</div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-800">{JSON.stringify(fieldValue, null, 2)}</div>
            </div>
          </div>
        )
    }
  }

  // If we have a schema, use it to render fields
  if (outputSchema?.properties) {
    const elements = Object.entries(outputSchema.properties).map(([fieldName, fieldDef]) => {
      const fieldValue = outputData[fieldName]
      return renderField(fieldName, fieldValue, fieldDef)
    }).filter(Boolean)

    if (elements.length > 0) {
      return <div className="space-y-4">{elements}</div>
    }
  }

  // Fallback: render all fields from outputData
  const elements = Object.entries(outputData).map(([fieldName, fieldValue]) => {
    return renderField(fieldName, fieldValue, null)
  }).filter(Boolean)

  if (elements.length > 0) {
    return <div className="space-y-4">{elements}</div>
  }

  // Final fallback: show JSON
  return (
    <div className="border border-gray-300 rounded-lg p-4 bg-white">
      <div className="mb-2 text-sm font-medium text-gray-700">Output Data</div>
      <div className="bg-gray-50 rounded-lg p-4">
        <pre className="text-xs text-gray-800 overflow-auto">
          {JSON.stringify(outputData, null, 2)}
        </pre>
      </div>
    </div>
  )
}
