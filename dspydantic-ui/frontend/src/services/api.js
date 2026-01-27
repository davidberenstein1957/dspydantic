import axios from 'axios'

// Use empty string for development (to use Vite proxy) or env var for production
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Store the error handler function
let errorHandler = null

// Setup error interceptor - call this with showError callback from ErrorContext
// This should be called once when the app initializes
export function setupErrorInterceptor(showError) {
  errorHandler = showError
}

// Set up the interceptor once (this runs when the module loads)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Show errors for HTTP errors and network errors
    if (errorHandler) {
      // Network errors (no response) or HTTP errors (with response)
      if (error.response || error.code === 'ERR_NETWORK' || error.message?.includes('Network Error')) {
        errorHandler(error)
      }
    }
    return Promise.reject(error)
  }
)

export const tasksApi = {
  list: () => api.get('/api/tasks'),
  get: (id) => api.get(`/api/tasks/${id}`),
  create: (data) => api.post('/api/tasks', data),
  update: (id, data) => api.put(`/api/tasks/${id}`, data),
  delete: (id) => api.delete(`/api/tasks/${id}`),
  validate: (id, data) => api.post(`/api/tasks/${id}/validate`, data),
  validateSchema: (id) => api.get(`/api/tasks/${id}/schema-validation`),
  duplicate: (id, data) => api.post(`/api/tasks/${id}/duplicate`, data),
}

export const examplesApi = {
  list: (taskId, params) => api.get(`/api/tasks/${taskId}/examples`, { params }),
  get: (id) => api.get(`/api/examples/${id}`),
  create: (taskId, data) => api.post(`/api/tasks/${taskId}/examples`, data),
  update: (id, data) => api.put(`/api/examples/${id}`, data),
  delete: (id) => api.delete(`/api/examples/${id}`),
  bulkImport: (taskId, data) => api.post(`/api/tasks/${taskId}/examples/import`, data),
  uploadFile: (taskId, file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/api/tasks/${taskId}/examples/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },
  updateStatus: (id, status) => api.patch(`/api/examples/${id}/status`, { status }),
  duplicate: (id, data) => api.post(`/api/examples/${id}/duplicate`, data),
  generateOutput: (id, data) => api.post(`/api/examples/${id}/generate-output`, data),
  bulkUpdateStatus: (data) => api.post('/api/examples/bulk/update-status', data),
  bulkGenerateOutput: (data) => api.post('/api/examples/bulk/generate-output', data),
}

export const optimizationApi = {
  trigger: (taskId, config) => api.post(`/api/tasks/${taskId}/optimize`, config),
  get: (runId) => api.get(`/api/optimization-runs/${runId}`),
  list: (taskId) => api.get(`/api/tasks/${taskId}/optimization-runs`),
  getDeleteInfo: (runId) => api.get(`/api/optimization-runs/${runId}/delete-info`),
  delete: (runId) => api.delete(`/api/optimization-runs/${runId}`),
}

export const promptsApi = {
  list: (taskId) => api.get(`/api/tasks/${taskId}/prompts`),
  get: (versionId) => api.get(`/api/prompts/${versionId}`),
  create: (taskId, data) => api.post(`/api/tasks/${taskId}/prompts`, data),
  update: (versionId, data) => api.put(`/api/prompts/${versionId}`, data),
  activate: (versionId) => api.post(`/api/prompts/${versionId}/activate`),
  compare: (versionId1, versionId2) => api.get('/api/prompts/compare', {
    params: { version_id_1: versionId1, version_id_2: versionId2 },
  }),
  getDeleteInfo: (versionId) => api.get(`/api/prompts/${versionId}/delete-info`),
  delete: (versionId) => api.delete(`/api/prompts/${versionId}`),
}

export const evaluationApi = {
  trigger: (taskId, config) => api.post(`/api/tasks/${taskId}/evaluate`, config),
  get: (runId) => api.get(`/api/evaluation-runs/${runId}`),
  list: (taskId) => api.get(`/api/tasks/${taskId}/evaluation-runs`),
  getResults: (runId) => api.get(`/api/evaluation-runs/${runId}/results`),
  retryExample: (runId, resultId) => api.post(`/api/evaluation-runs/${runId}/results/${resultId}/retry`),
  delete: (runId) => api.delete(`/api/evaluation-runs/${runId}`),
}

export default api
