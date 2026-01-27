export default function ExampleFilters({ filters, onFilterChange, onClearFilters }) {
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 mb-4">
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Input Complete
          </label>
          <select
            value={filters.input_complete === null || filters.input_complete === undefined ? '' : filters.input_complete}
            onChange={(e) =>
              onFilterChange(
                'input_complete',
                e.target.value === '' ? null : e.target.value === 'true'
              )
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="">All</option>
            <option value="true">Complete</option>
            <option value="false">Incomplete</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Output Complete
          </label>
          <select
            value={filters.output_complete === null || filters.output_complete === undefined ? '' : filters.output_complete}
            onChange={(e) =>
              onFilterChange(
                'output_complete',
                e.target.value === '' ? null : e.target.value === 'true'
              )
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="">All</option>
            <option value="true">Complete</option>
            <option value="false">Incomplete</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Status
          </label>
          <select
            value={filters.status || ''}
            onChange={(e) => onFilterChange('status', e.target.value || null)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="">All</option>
            <option value="">No Status</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
            <option value="pending">Pending</option>
            <option value="reviewed">Reviewed</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Created After
          </label>
          <input
            type="date"
            value={filters.created_after}
            onChange={(e) => onFilterChange('created_after', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Created Before
          </label>
          <input
            type="date"
            value={filters.created_before}
            onChange={(e) => onFilterChange('created_before', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
        </div>
      </div>

      {onClearFilters && (
        <div className="mt-4 flex justify-end">
          <button
            onClick={onClearFilters}
            className="px-4 py-2 text-sm text-gray-700 hover:text-gray-900"
          >
            Clear Filters
          </button>
        </div>
      )}
    </div>
  )
}
