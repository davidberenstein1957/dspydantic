import { useState, useEffect, useRef } from 'react'

export default function ErrorNotification({ error, onDismiss }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [copied, setCopied] = useState(false)
  const timerRef = useRef(null)

  const resetTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
    }
    if (error) {
      timerRef.current = setTimeout(() => {
        onDismiss()
      }, 5000)
    }
  }

  useEffect(() => {
    if (error) {
      resetTimer()
    }
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
      }
    }
  }, [error, onDismiss])

  const handleMouseEnter = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }

  const handleMouseLeave = () => {
    resetTimer()
  }

  const handleCopyError = async () => {
    try {
      const errorText = formatErrorForCopy(error)
      await navigator.clipboard.writeText(errorText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy error:', err)
    }
  }

  const formatErrorForCopy = (err) => {
    const parts = []
    
    if (err.response) {
      parts.push(`Status: ${err.response.status} ${err.response.statusText || ''}`)
      parts.push(`URL: ${err.config?.url || err.request?.responseURL || 'Unknown'}`)
      parts.push(`Method: ${err.config?.method?.toUpperCase() || 'Unknown'}`)
      parts.push('')
      parts.push('Response Data:')
      parts.push(JSON.stringify(err.response.data, null, 2))
    } else {
      parts.push(`Error: ${err.message || 'Unknown error'}`)
      if (err.code) {
        parts.push(`Code: ${err.code}`)
      }
    }
    
    parts.push('')
    parts.push('Full Error Object:')
    parts.push(JSON.stringify(err, Object.getOwnPropertyNames(err), 2))
    
    if (err.stack) {
      parts.push('')
      parts.push('Stack Trace:')
      parts.push(err.stack)
    }
    
    return parts.join('\n')
  }

  if (!error) return null

  const errorMessage = error.response?.data?.detail || error.message || 'An error occurred'
  const errorStatus = error.response?.status
  const fullError = error.response?.data || error

  return (
    <div 
      className="fixed bottom-4 right-4 z-50 max-w-md"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="bg-red-50 border border-red-200 rounded-lg shadow-lg overflow-hidden">
        <div className="flex items-start justify-between p-3">
          <div className="flex items-start flex-1 min-w-0">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-red-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
            <div className="ml-3 flex-1 min-w-0">
              <p className="text-sm font-medium text-red-800">
                {errorStatus ? `Error ${errorStatus}` : 'Error'}
              </p>
              <p className="text-sm text-red-700 mt-1 break-words">
                {typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage)}
              </p>
            </div>
          </div>
          <div className="flex items-start ml-2 space-x-1">
            <button
              onClick={handleCopyError}
              className="text-red-600 hover:text-red-800 text-xs px-2 py-1 rounded hover:bg-red-100 flex items-center"
              title="Copy error details"
            >
              {copied ? (
                <span className="text-green-600">✓</span>
              ) : (
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
              )}
            </button>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-red-600 hover:text-red-800 text-xs px-2 py-1 rounded hover:bg-red-100"
              title={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? '▼' : '▶'}
            </button>
            <button
              onClick={onDismiss}
              className="text-red-600 hover:text-red-800 text-xs px-2 py-1 rounded hover:bg-red-100"
              title="Dismiss"
            >
              ×
            </button>
          </div>
        </div>
        {isExpanded && (
          <div className="px-3 pb-3 border-t border-red-200">
            <div className="mt-2">
              <p className="text-xs font-medium text-red-800 mb-1">Full Error Details:</p>
              <pre className="text-xs text-red-700 bg-red-100 p-2 rounded overflow-auto max-h-64">
                {JSON.stringify(fullError, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
