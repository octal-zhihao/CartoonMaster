<script setup lang="ts">

// 按钮，多选框的导入
import {Avatar, Menu as IconMenu, WarnTriangleFilled,} from '@element-plus/icons-vue'
import {computed, onMounted, ref, watch} from 'vue'
// 展示预测结果的导入
import {ElIcon, ElImage, ElMessage} from 'element-plus'


// 上传图片的导入
import axios from 'axios'

// 进度条相关变量
const progressBarStatus = ref("close")
const loadingTime = ref(5)
const percentage2 = ref(0)
const progressBarColors = [
  {color: '#f56c6c', percentage: 10},
  {color: '#f57c6c', percentage: 15},
  {color: '#f58c6c', percentage: 20},
  {color: '#e6973c', percentage: 25},
  {color: '#e6a23c', percentage: 30},
  {color: '#e6cf3c', percentage: 35},
  {color: '#d5e63c', percentage: 40},
  {color: '#a2e63c', percentage: 45},
  {color: '#53e63c', percentage: 50},
  {color: '#5cb85e', percentage: 55},
  {color: '#5cb87a', percentage: 60},
  {color: '#5cb8a7', percentage: 65},
  {color: '#5cb0b8', percentage: 70},
  {color: '#19e0fa', percentage: 80},
  {color: '#19befa', percentage: 85},
  {color: '#1973fa', percentage: 90},
  {color: '#545cff', percentage: 95},
  {color: '#6f7ad3', percentage: 100},
]
const progressBarPause = ref(true)
const isLoadProgress = computed(() => progressBarStatus.value)


// 预测结果显示的具体实现
const pre_result_img_urls = ref<string[]>([])

const img_len = ref(0)
const view_len = ref(0)
const ganFinImgCount = ref(1)
const ganGenerateImgCount = ref(4)
const ddpmGenerateImgCount = ref(1)
setListLength(pre_result_img_urls);

function sqrtAndCeil(x: number): number {
  // 首先计算平方根
  const sqrtValue = Math.sqrt(x);
  // 然后使用 Math.ceil 向上取整
  return Math.ceil(sqrtValue);
}

function setListLength(list: any) {

  let sp_number;
  sp_number = sqrtAndCeil(list.length)

  img_len.value = 25 / sp_number
  view_len.value = 26

}


// 进度条伪加载
onMounted(() => {
  setInterval(() => {
    if (percentage2.value <= 97) {
      percentage2.value = (percentage2.value % 100) + 1
    } else {
      if (progressBarPause.value) {
        percentage2.value = 98
      } else {
        percentage2.value = 100
      }
    }
  }, loadingTime.value * 10)
})


// 多选框的具体实现
const checkAll = ref(false)
const indeterminate = ref(false)
const value = ref("DC_GAN")//这个value绑定了选择框的选中值
const models = ref([
  {
    value: 'DC_GAN',
    label: 'DC_GAN',
  },
  {
    value: 'DDPM',
    label: 'DDPM',
  },
  {
    value: 'WGAN',
    label: 'WGAN',
  },
  {
    value: 'WGAN-GP',
    label: 'WGAN-GP',
  },
  {
    value: 'GAN',
    label: 'GAN',
  }
])

const selectedModelLayout = computed(() => value.value)

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

  const formData = new FormData()

  if (ganGenerateImgCount.value < ganFinImgCount.value) {
    ElMessage.warning("生成总数应大于最终图片数，已将生成总数设置为最终图片数")
    ganGenerateImgCount.value = ganFinImgCount.value
  }

  isInferencing.value = true
  progressBarStatus.value = "open"
  percentage2.value = 0

  try {

    formData.append('model', value.value)
    formData.append('gen_search_num', ganGenerateImgCount.value)
    formData.append('gen_num', ganFinImgCount.value)
    formData.append('ddpm_num', ddpmGenerateImgCount.value)


    const response = await axios.post('/api/generate', formData, {
      headers: {'Content-Type': 'multipart/form-data'},
    })
    console.log(response)

    if (response.status === 200) {
      ElMessage.success('推理完成')
      progressBarPause.value = false
      // pre_result_img_urls.value = response.data.result_images

      pre_result_img_urls.value = [
        ...response.data.res,
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
    percentage2.value = 100
    progressBarStatus.value = "close"
    progressBarPause.value = true
  }
}

</script>

<template>
  <div class="common-layout">
    <el-container class="main-container" style="height: 100vh;">
      <el-header height="55px" style="border-bottom: 1px solid var(--el-border-color)">
        <div style="width: 100%; height: 100%;display: inline-flex;align-items: center;justify-content: center;">
          <el-icon :size="25" color="#606266">
            <Avatar/>
          </el-icon>
          <el-text size="large" tag="b">基于gan，ddpm的卡通图象生成-demo</el-text>
        </div>
      </el-header>
      <el-container height="100vh">

        <!-- 侧边栏 -->
        <el-aside width="15vw" style="border-right: 1px solid var(--el-border-color);">
          <el-menu>
            <el-menu-item index="2">
              <el-icon>
                <icon-menu/>
              </el-icon>
              <template #title>头像生成</template>
            </el-menu-item>
            <el-menu-item index="3" disabled>
              <el-icon>
                <WarnTriangleFilled/>
              </el-icon>
              <template #title>施工中</template>
            </el-menu-item>
            <el-menu-item index="4" disabled>
              <el-icon>
                <WarnTriangleFilled/>
              </el-icon>
              <template #title>施工中</template>
            </el-menu-item>
          </el-menu>
        </el-aside>

        <!-- 主体部分 -->
        <el-main>
          <div style="height: 2vw;width: 100%">
            <!--模型选择-->
            <el-select v-model="value" placeholder="Select" style="width: 240px">
              <el-option
                  v-for="item in models"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value"
              />
            </el-select>
            <el-text size="large"><-模型选择（默认DC_GAN）</el-text>
          </div>
          <div style="height: 0.5vw;width: 100%"></div>
          <div style="height: 0.5vw;width: 100%;border-top: 2px dashed rgb(197.7, 225.9, 255)"></div>
          <el-row>
            <el-col :span="11">
              <el-row align="top" :gutter="10">
                <el-col :span="24" justify="center">
                  <div v-if="selectedModelLayout === 'DC_GAN' || selectedModelLayout === 'WGAN' || selectedModelLayout === 'WGAN-GP' || selectedModelLayout === 'GAN'">
                    <div class="Gan-sliders">
                      <!--                    <el-text size="large">最终图片数</el-text>-->
                      <div style="color: rgb(51.2, 126.4, 204)">最终图片数</div>
                      <el-slider class="fin-img-count" v-model="ganFinImgCount" show-input :min="1" :max="100"/>
                      <div style="color: rgb(51.2, 126.4, 204)">生成总数</div>
                      <el-slider class="fin-img-count" v-model="ganGenerateImgCount" show-input :min="1" :max="100"/>
                    </div>
                  </div>
                  <div v-else-if="selectedModelLayout === 'DDPM'">
                    <div class="DDPM-sliders">
                      <div style="color: rgb(51.2, 126.4, 204)">生成图片数量</div>
                      <el-slider class="fin-img-count" v-model="ddpmGenerateImgCount" show-input :min="1" :max="100"/>
                      <div style="color: rgb(51.2, 126.4, 204)">施工中</div>
                      <el-slider class="fin-img-count" v-model="ganGenerateImgCount" show-input :min="1" :max="100"
                                 :disabled="true"/>
                    </div>
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
                  <div style="height: 3vw;">
                    <div class="demo-progress" style="margin-top: 1.2vw; width: 25vw;margin-left: 8vw">
                      <el-progress :percentage="percentage2" :color="progressBarColors" striped striped-flow
                                   v-if="isLoadProgress === 'open'" :stroke-width="15"/>
                    </div>
                  </div>
                </el-col>
                <el-col :span="24" align="middle">
                  <!-- 推理结果展示 -->
                  <div class="pre_result_show"
                       :style="{ width: view_len+1 + 'vw', height: view_len + 'vw', overflowY: 'auto' }"
                       style="border: 2px dashed rgb(159.5, 206.5, 255);border-radius: 6px;">
                    <el-image :style="{ width: img_len + 'vw', height: img_len+0.2 + 'vw' }"
                              v-for="(url, index) in pre_result_img_urls"
                              :key="url" :src="url" :zoom-rate="1.2" :max-scale="7" :min-scale="0.2"
                              :preview-src-list="pre_result_img_urls"
                              :initial-index=index fit="cover"/>
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
.Gan-sliders {
  max-width: 32vw;
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  flex-direction: column;

}

.Gan-sliders .el-slider {
  margin-top: 0;
  margin-left: 12px;
}
</style>