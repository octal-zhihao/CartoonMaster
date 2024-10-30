from flask import Blueprint
api = Blueprint('api', __name__, url_prefix='/api')

# import写在最后防止循环导入
from Web.back_end.api import generate