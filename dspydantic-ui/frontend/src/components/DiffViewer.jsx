import { useMemo } from 'react'

function diffLines(oldText, newText) {
  const oldLines = oldText.split('\n')
  const newLines = newText.split('\n')
  const result = []
  
  let i = 0
  let j = 0
  
  while (i < oldLines.length || j < newLines.length) {
    if (i >= oldLines.length) {
      result.push({ type: 'added', line: newLines[j], oldLineNum: null, newLineNum: j + 1 })
      j++
    } else if (j >= newLines.length) {
      result.push({ type: 'removed', line: oldLines[i], oldLineNum: i + 1, newLineNum: null })
      i++
    } else if (oldLines[i] === newLines[j]) {
      result.push({ type: 'unchanged', line: oldLines[i], oldLineNum: i + 1, newLineNum: j + 1 })
      i++
      j++
    } else {
      // Check if line was moved
      const nextOldIndex = oldLines.slice(i + 1).indexOf(newLines[j])
      const nextNewIndex = newLines.slice(j + 1).indexOf(oldLines[i])
      
      if (nextOldIndex !== -1 && (nextNewIndex === -1 || nextOldIndex < nextNewIndex)) {
        // Line was removed
        result.push({ type: 'removed', line: oldLines[i], oldLineNum: i + 1, newLineNum: null })
        i++
      } else if (nextNewIndex !== -1) {
        // Line was added
        result.push({ type: 'added', line: newLines[j], oldLineNum: null, newLineNum: j + 1 })
        j++
      } else {
        // Line was modified
        result.push({ type: 'removed', line: oldLines[i], oldLineNum: i + 1, newLineNum: null })
        result.push({ type: 'added', line: newLines[j], oldLineNum: null, newLineNum: j + 1 })
        i++
        j++
      }
    }
  }
  
  return result
}

export default function DiffViewer({ oldText, newText, title }) {
  const diff = useMemo(() => diffLines(oldText || '', newText || ''), [oldText, newText])

  return (
    <div className="border rounded-lg overflow-hidden">
      {title && (
        <div className="bg-gray-100 px-4 py-2 border-b">
          <h4 className="text-sm font-medium text-gray-900">{title}</h4>
        </div>
      )}
      <div className="overflow-auto max-h-96 font-mono text-xs">
        <table className="w-full border-collapse">
          <tbody>
            {diff.map((item, index) => (
              <tr
                key={index}
                className={
                  item.type === 'added'
                    ? 'bg-green-50'
                    : item.type === 'removed'
                    ? 'bg-red-50'
                    : 'bg-white'
                }
              >
                <td className="px-2 py-1 text-gray-500 border-r border-gray-200 text-right w-12">
                  {item.oldLineNum || ''}
                </td>
                <td className="px-2 py-1 text-gray-500 border-r border-gray-200 text-right w-12">
                  {item.newLineNum || ''}
                </td>
                <td className="px-3 py-1">
                  <span
                    className={
                      item.type === 'added'
                        ? 'text-green-800'
                        : item.type === 'removed'
                        ? 'text-red-800 line-through'
                        : 'text-gray-800'
                    }
                  >
                    {item.type === 'added' && '+'}
                    {item.type === 'removed' && '-'}
                    {item.type === 'unchanged' && ' '}
                    {' '}
                    {item.line || '\u00A0'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
