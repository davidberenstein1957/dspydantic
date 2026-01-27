export default function OptimizationRunFilters({ filters, onFilterChange, onClearFilters }) {
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 mb-4">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
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
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Started After
          </label>
          <input
            type="date"
            value={filters.started_after || ''}
            onChange={(e) => onFilterChange('started_after', e.target.value || '')}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Started Before
          </label>
          <input
            type="date"
            value={filters.started_before || ''}
            onChange={(e) => onFilterChange('started_before', e.target.value || '')}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Optimizer
          </label>
          <select
            value={filters.optimizer || ''}
            onChange={(e) => onFilterChange('optimizer', e.target.value || null)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="">All</option>
            <option value="bootstrap">Bootstrap</option>
            <option value="miprompts">MIPrompts</option>
            <option value="miprompts_legacy">MIPrompts (Legacy)</option>
          </select>
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
