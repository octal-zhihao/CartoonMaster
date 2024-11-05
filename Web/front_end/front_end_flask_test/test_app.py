from flask import Flask, request, Response, jsonify
import json
from flask_cors import CORS
import os
import time
import random

# 设置当前工作目录为文件所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)  # 允许跨域请求

test_time = round(random.uniform(5, 8), 2)


def count_images_in_directory(directory):
    image_extensions = ('.jpg', '.jpeg', '.png')
    files = os.listdir(directory)
    image_count = sum(1 for file in files if file.lower().endswith(image_extensions))
    return image_count


@app.route('/api/generate', methods=['POST'])
def test_generate():
    data = request.form
    all_test_img_count = count_images_in_directory("./test_static")
    res = []
    if "GAN" in data.get("model"):
        time.sleep(test_time)  # 模拟生成图片耗时
        test_img_count = int(data.get("gen_num"))
        if test_img_count > all_test_img_count:
            test_img_count = all_test_img_count
        unique_numbers = random.sample(range(1, all_test_img_count + 1), test_img_count)
        for i in unique_numbers:
            res.append("./front_end_flask_test/test_static"+"/test (" + str(i) + ").jpg")
    elif "DDPM" in data.get("model"):
        time.sleep(test_time*2)  # 模拟生成图片耗时
        test_img_count = int(data.get("ddpm_num"))
        if test_img_count > all_test_img_count:
            test_img_count = all_test_img_count
        unique_numbers = random.sample(range(1, all_test_img_count + 1), test_img_count)
        for i in unique_numbers:
            res.append("./front_end_flask_test/test_static"+"/test (" + str(i) + ").jpg")
    result = {
        "res": res
    }
    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
