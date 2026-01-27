export default function TaskFilters({ filters, onFilterChange, onClearFilters }) {
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 mb-4">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Search
          </label>
          <input
            type="text"
            placeholder="Search tasks..."
            value={filters.search || ''}
            onChange={(e) => onFilterChange('search', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Has Examples
          </label>
          <select
            value={filters.has_examples === null ? '' : filters.has_examples}
            onChange={(e) =>
              onFilterChange(
                'has_examples',
                e.target.value === '' ? null : e.target.value === 'true'
              )
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
          >
            <option value="">All</option>
            <option value="true">Has Examples</option>
            <option value="false">No Examples</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Created After
          </label>
          <input
            type="date"
            value={filters.created_after || ''}
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
            value={filters.created_before || ''}
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
