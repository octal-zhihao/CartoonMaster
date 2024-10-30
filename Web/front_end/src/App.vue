<script setup lang="ts">

// 按钮，多选框的导入
import {
  Menu as IconMenu,
  WarnTriangleFilled,
  Avatar,
} from '@element-plus/icons-vue'
import { ref, watch, onMounted } from 'vue'
import type { CheckboxValueType, UploadFile } from 'element-plus'


// 上传图片的导入
import axios from 'axios'
import { ElMessage, ElUpload, ElIcon } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'


// 展示预测结果的导入
import { ElImage } from 'element-plus';


// 预测结果显示的具体实现
const pre_result_img_urls = ref<string[]>([])

  const img_len = ref(0)
  const veiw_len = ref(0)
  setListLength(pre_result_img_urls);
  function setListLength(list: any) {
    if (list.length <= 1){
      img_len.value = 25
      veiw_len.value = 26
    }
    else if (list.length <= 4 && list.length >= 2) {
      img_len.value = 12.5
      veiw_len.value = 26
    }
    else {
      img_len.value = 8.33
      veiw_len.value = 26
    }
}


// 获取模型名称列表的请求
const fetchModels = async () => {
  try {
    const response = await axios.get('/api/models')
    if (response.status === 200) {
      models.value = response.data.map((model: string) => ({
        value: model,
        label: model,
      }))
      ElMessage.success('模型列表已更新')
    }
  } catch (error) {
    ElMessage.error('获取模型列表失败，使用默认列表')
  }
}


// 上传图片到服务器的具体实现
const imageUrl = ref('')
const uped_img_local_path = ref('')


// 多选框的具体实现
const checkAll = ref(false)
const indeterminate = ref(false)
const value = ref<CheckboxValueType[]>([])//这个value绑定了多选框的选中值
const models = ref([
  {
    value: 'U-net',
    label: 'U-net',
  },
  {
    value: 'DeepLab',
    label: 'DeepLab',
  },
  {
    value: 'WeClip',
    label: 'WeClip',
  },
])
watch(value, (val) => {
  if (val.length === 0) {
    checkAll.value = false
    indeterminate.value = false
  } else if (val.length === models.value.length) {
    checkAll.value = true
    indeterminate.value = false
  } else {
    indeterminate.value = true
  }
})
// 触发推理请求
const isInferencing = ref(false) // 控制推理按钮的状态

const startInference = async () => {
  if (!imageUrl.value || value.value.length === 0) {
    ElMessage.error('请上传图片并选择模型')
    return
  }

  isInferencing.value = true

  try {
    const response = await axios.post('/api/start', {
      image_url: uped_img_local_path.value,
      models: value.value,
    })
    console.log(response)

    if (response.status === 200) {
      ElMessage.success('推理完成')
      // pre_result_img_urls.value = response.data.result_images

      pre_result_img_urls.value = [
        ...response.data.Unet,
        ...response.data.deeplab,
        ...response.data.WeClip
      ]
      console.log(pre_result_img_urls.value)
      setListLength(pre_result_img_urls.value)
    } else {
      ElMessage.error('推理失败')
    }
  } catch (error) {
    ElMessage.error(`推理失败: ${error}`)
  } finally {
    isInferencing.value = false
  }
}

// 初始化时获取模型名称列表
onMounted(() => {
  fetchModels()
})


</script>

<template>
  <div class="common-layout">
    <el-container class="main-container" style="height: 100vh;">
      <el-header height="55px" style="border-bottom: 1px solid var(--el-border-color)">
        <div style="width: 100%; height: 100%;display: inline-flex;align-items: center;justify-content: center;">
          <el-icon :size="25" color="#606266">
            <Avatar />
          </el-icon>
          <el-text size="large" tag="b">基于gan，ddpm的卡通图象生成-demo</el-text>
        </div>
      </el-header>
      <el-container height="100vh">

        <!-- 侧边栏 -->
        <el-aside width="15vw" style="border-right: 1px solid var(--el-border-color);">
          <el-menu>
            <el-menu-item index="2">
              <el-icon><icon-menu /></el-icon>
              <template #title>头像生成</template>
            </el-menu-item>
            <el-menu-item index="3" disabled>
              <el-icon>
                <WarnTriangleFilled />
              </el-icon>
              <template #title>施工中</template>
            </el-menu-item>
            <el-menu-item index="4" disabled>
              <el-icon>
                <WarnTriangleFilled />
              </el-icon>
              <template #title>施工中</template>
            </el-menu-item>
          </el-menu>
        </el-aside>

        <!-- 主体部分 -->
        <el-main>
          <el-row>
            <el-col :span="11">
              <el-row align="top" :gutter="10">
                <el-col :span="24" align="middle" justify="center">
                  <!-- 选择模型部分 -->
                  <div>
                  </div>
                </el-col>
                <el-col>
                  <div style="height:5vw;"></div>
                </el-col>
                <el-col :span="24" align="middle">
                  <!-- 上传图片部分 -->
                  <div style="width: 95%;">
                  </div>
                </el-col>

              </el-row>
            </el-col>
            <el-col :span="2"></el-col>
            <el-col :span="11">
              <el-row align="top" :gutter="10">
                <el-col :span="24" align="middle">
                  <el-button 
                  :type="isInferencing ? 'info' : 'primary'" 
                  plain 
                  :disabled="isInferencing"
                  @click="startInference"
                  style="width: 30vw;height: 4vw;">
                  {{ isInferencing ? '推理中' : '开始推理' }}
                </el-button>
                </el-col>
                <el-col :span="24">
                  <div style="height: 3vw;"></div>
                </el-col>
                <el-col :span="24" align="middle" >
                  <!-- 推理结果展示 -->
                  <div class="pre_result_show"
                    :style="{ width: veiw_len+1 + 'vw', height: veiw_len + 'vw', overflowY: 'auto' }" style="border: 2px dashed rgb(159.5, 206.5, 255);border-radius: 6px;">
                    <el-image :style="{ width: img_len + 'vw', height: img_len+0.2 + 'vw' }" v-for="(url, index) in pre_result_img_urls"
                      :key="url" :src="url" :zoom-rate="1.2" :max-scale="7" :min-scale="0.2" :preview-src-list="pre_result_img_urls"
                      :initial-index=index fit="cover" />
                  </div>
                </el-col>
              </el-row>
            </el-col>
          </el-row>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<style scoped>

</style>