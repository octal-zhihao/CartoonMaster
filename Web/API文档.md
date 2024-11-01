# api

## Generate(使用指定模型生成图像，接收表单数据，返回图片路径)

### url
- /api/generate

### method
- POST

### args
|  args          | required | request type | type |  remarks                  |
|----------------|----------|--------------|------|---------------------------|
| model          |  true    |    form      | str  | 使用的模型  |
| gen_search_num |  false   |    form      | int  | 总生成数  |
| gen_num        |  false   |    form      | int  | 最终图片数 |
| gen_mean       |  false   |    form      | int  | 噪声均值（暂无）   |
| gen_std        |  false   |    form      | int  | 噪声方差（暂无）   |

### request
```
{
    "model": "DC_GAN",
    "gen_search_num": 10,
    "gen_num": 5,
    "gen_mean": 0,
    "gen_std": 1
}
```

### return
```json
{
    "res": "[./static/gan_img.png]"
}
```




