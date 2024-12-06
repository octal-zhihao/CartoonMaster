import sys
import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, 'training'))
sys.path.append(os.path.join(current_dir, 'training', 'DDPM'))
# 可以的话改一下引用包的路径，这样就不需要这几行了↑


from Web.back_end import app

if __name__ == '__main__':
    app.run()
