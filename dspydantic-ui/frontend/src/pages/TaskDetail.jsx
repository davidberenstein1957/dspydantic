import { useState, useEffect } from 'react'
import { useParams, useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { tasksApi } from '../services/api'
import ExamplesTab from '../components/ExamplesTab'
import OptimizationTab from '../components/OptimizationTab'
import EvaluationTab from '../components/EvaluationTab'
import PromptsTab from '../components/PromptsTab'
import TaskSettingsTab from '../components/TaskSettingsTab'

export default function TaskDetail() {
  const { id } = useParams()
  const location = useLocation()
  const [activeTab, setActiveTab] = useState(location.state?.activeTab || 'examples')

  // Update activeTab when location state changes
  useEffect(() => {
    if (location.state?.activeTab) {
      setActiveTab(location.state.activeTab)
    }
  }, [location.state])

  const { data: task, isLoading } = useQuery({
    queryKey: ['tasks', id],
    queryFn: async () => {
      const response = await tasksApi.get(id)
      return response.data
    },
  })

  if (isLoading) {
    return <div className="text-center py-12">Loading task...</div>
  }

  if (!task) {
    return <div className="text-center py-12">Task not found</div>
  }

  const tabs = [
    { id: 'examples', name: 'Examples', count: task.example_count },
    { id: 'optimization', name: 'Optimization' },
    { id: 'evaluation', name: 'Evaluation' },
    { id: 'prompts', name: 'Prompts' },
    { id: 'settings', name: 'Settings' },
  ]

  return (
    <div className="w-full px-4 sm:px-6 lg:px-8">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">{task.name}</h1>
        {task.description && (
          <p className="mt-2 text-sm text-gray-700">{task.description}</p>
        )}
      </div>

      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.name}
              {tab.count !== undefined && (
                <span className="ml-2 bg-gray-100 text-gray-900 py-0.5 px-2.5 rounded-full text-xs">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      <div className="mt-6">
        {activeTab === 'examples' && <ExamplesTab taskId={parseInt(id)} />}
        {activeTab === 'optimization' && <OptimizationTab taskId={parseInt(id)} task={task} />}
        {activeTab === 'evaluation' && <EvaluationTab taskId={parseInt(id)} task={task} />}
        {activeTab === 'prompts' && <PromptsTab taskId={parseInt(id)} />}
        {activeTab === 'settings' && <TaskSettingsTab task={task} />}
      </div>
    </div>
  )
}
