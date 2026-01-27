import ReactMarkdown from 'react-markdown'

export default function ContentViewer({ inputData, inputSchema }) {
  if (!inputData || (typeof inputData !== 'object')) {
    return <div className="text-gray-500">No input data available</div>
  }

  const renderContent = () => {
    const contentElements = []
    
    // Handle images array (from PDFs or multiple images) - check this first
    if (inputData.images && Array.isArray(inputData.images) && inputData.images.length > 0) {
      inputData.images.forEach((image, idx) => {
        if (!image) return
        
        // Check if it's base64 (either data URI or raw base64 string)
        const isDataUri = typeof image === 'string' && image.startsWith('data:')
        const isUrl = typeof image === 'string' && (image.startsWith('http://') || image.startsWith('https://'))
        const isBase64 = typeof image === 'string' && !isDataUri && !isUrl && image.length > 100
        const imageSrc = isDataUri 
          ? image 
          : isBase64 
            ? `data:image/png;base64,${image}`
            : image

        contentElements.push(
          <div key={`image-${idx}`} className="border border-gray-300 rounded-lg p-4 bg-white max-w-full overflow-hidden">
            <div className="mb-2 text-sm font-medium text-gray-700">
              Image {idx + 1} {inputData.images.length > 1 && `of ${inputData.images.length}`}
            </div>
            <div className="flex justify-center bg-gray-50 rounded-lg p-4 overflow-hidden">
              <img
                src={imageSrc}
                alt={`Input image ${idx + 1}`}
                className="max-w-full max-h-[600px] object-contain rounded shadow-sm"
                style={{ width: 'auto', height: 'auto' }}
                onError={(e) => {
                  console.error('Failed to load image:', imageSrc.substring(0, 50))
                  e.target.style.display = 'none'
                  e.target.parentElement.innerHTML = '<div class="text-red-500 text-sm">Failed to load image</div>'
                }}
              />
            </div>
          </div>
        )
      })
    }

    // Handle individual image/pdf fields from schema
    if (inputSchema?.properties) {
      const imageFields = Object.entries(inputSchema.properties).filter(
        ([_, def]) => def.type === 'image' || def.type === 'pdf'
      )

      imageFields.forEach(([fieldName, fieldDef]) => {
        const fieldValue = inputData[fieldName]
        if (!fieldValue) return

        const isDataUri = typeof fieldValue === 'string' && fieldValue.startsWith('data:')
        const isUrl = typeof fieldValue === 'string' && (fieldValue.startsWith('http://') || fieldValue.startsWith('https://'))
        const isBase64 = typeof fieldValue === 'string' && !isDataUri && !isUrl && fieldValue.length > 100
        const isPdf = fieldDef.type === 'pdf' || (isDataUri && fieldValue.includes('application/pdf'))
        
        if (isPdf) {
          const pdfSrc = isDataUri
            ? fieldValue
            : isBase64
              ? `data:application/pdf;base64,${fieldValue}`
              : fieldValue

          contentElements.push(
            <div key={fieldName} className="border border-gray-300 rounded-lg p-4 bg-white max-w-full overflow-hidden">
              <div className="mb-2 text-sm font-medium text-gray-700">{fieldName}</div>
              <div className="bg-gray-50 rounded-lg p-4 overflow-hidden">
                <iframe
                  src={pdfSrc}
                  className="w-full h-[600px] border-0 rounded max-w-full"
                  title={fieldName}
                />
              </div>
            </div>
          )
        } else {
          const imageSrc = isDataUri
            ? fieldValue
            : isBase64
              ? `data:image/png;base64,${fieldValue}`
              : fieldValue

          contentElements.push(
            <div key={fieldName} className="border border-gray-300 rounded-lg p-4 bg-white max-w-full overflow-hidden">
              <div className="mb-2 text-sm font-medium text-gray-700">{fieldName}</div>
              <div className="flex justify-center bg-gray-50 rounded-lg p-4 overflow-hidden">
                <img
                  src={imageSrc}
                  alt={fieldName}
                  className="max-w-full max-h-[600px] object-contain rounded shadow-sm"
                  style={{ width: 'auto', height: 'auto' }}
                  onError={(e) => {
                    console.error('Failed to load image:', imageSrc.substring(0, 50))
                    e.target.style.display = 'none'
                    e.target.parentElement.innerHTML = '<div class="text-red-500 text-sm">Failed to load image</div>'
                  }}
                />
              </div>
            </div>
          )
        }
      })

      // Handle string fields from schema
      const stringFields = Object.entries(inputSchema.properties).filter(
        ([_, def]) => def.type === 'string'
      )

      stringFields.forEach(([fieldName, fieldDef]) => {
        const fieldValue = inputData[fieldName]
        if (fieldValue === undefined || fieldValue === null) return

        const shouldRenderMarkdown = fieldDef?.render_markdown !== false

        contentElements.push(
          <div key={fieldName} className="border border-gray-300 rounded-lg p-4 bg-white max-w-full overflow-hidden">
            <div className="mb-2 text-sm font-medium text-gray-700">{fieldName}</div>
            <div className="bg-gray-50 rounded-lg p-4 prose prose-sm max-w-none overflow-x-auto">
              {shouldRenderMarkdown ? (
                <ReactMarkdown>{String(fieldValue)}</ReactMarkdown>
              ) : (
                <div className="text-sm text-gray-800 whitespace-pre-wrap">{String(fieldValue)}</div>
              )}
            </div>
          </div>
        )
      })
    }

    // Handle text content (legacy field, always render as markdown)
    if (inputData.text) {
      contentElements.push(
        <div key="text" className="border border-gray-300 rounded-lg p-4 bg-white max-w-full overflow-hidden">
          <div className="mb-2 text-sm font-medium text-gray-700">Text Input</div>
          <div className="bg-gray-50 rounded-lg p-4 prose prose-sm max-w-none overflow-x-auto">
            <ReactMarkdown>{String(inputData.text)}</ReactMarkdown>
          </div>
        </div>
      )
    }

    // If we have content elements, return them
    if (contentElements.length > 0) {
      return <div className="space-y-4 max-w-full">{contentElements}</div>
    }

    // Fallback: show JSON with string fields as markdown
    return (
      <div className="border border-gray-300 rounded-lg p-4 bg-white max-w-full overflow-hidden">
        <div className="mb-2 text-sm font-medium text-gray-700">Input Data</div>
        <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
          <pre className="text-xs text-gray-800 whitespace-pre-wrap break-words">
            {JSON.stringify(inputData, null, 2)}
          </pre>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full max-w-full overflow-x-hidden">
      {renderContent()}
    </div>
  )
}
