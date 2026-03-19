/**
 * 临时存储待上传的文件和需求
 * 用于首页点击启动引擎后立即跳转，在Process页面再进行API调用
 */
import { reactive } from 'vue'

const state = reactive({
  files: [],
  simulationRequirement: '',
  marketQuestion: '',
  isPending: false
})

export function setPendingUpload(files, requirement, marketQuestion = '') {
  state.files = files
  state.simulationRequirement = requirement
  state.marketQuestion = marketQuestion
  state.isPending = true
}

export function getPendingUpload() {
  return {
    files: state.files,
    simulationRequirement: state.simulationRequirement,
    marketQuestion: state.marketQuestion,
    isPending: state.isPending
  }
}

export function clearPendingUpload() {
  state.files = []
  state.simulationRequirement = ''
  state.marketQuestion = ''
  state.isPending = false
}

export default state
