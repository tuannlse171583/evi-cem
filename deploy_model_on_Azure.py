!pip install azureml-sdk
!pip install pytorch-lightning
import os
import json
import requests

from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
import sys
sys.path.append('/content/utils.py')
sys.path.append('/content/clm.py')
sys.path.append('/content/evi_clm.py')
sys.path.append('/content/evi_cem.py')
!python /content/score.py
from azureml.core import Workspace

ws = Workspace.from_config()
# tải mô hình lên azure ml:
from azureml.core.model import Model

model = Model.register(
    workspace=ws,
    model_name="evi-cem-model",  # Tên mô hình trên Azure
    model_path="/content/epoch=49-step=5200.ckpt",  # Đường dẫn tệp mô hình đã lưu
    description="Mô hình nhận diện dựa trên PyTorch"
)

print(f"Model registered: {model.name} - Version: {model.version}")
registered_model = Model(ws, "evi-cem-model", version=5)

from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.runconfig import DockerConfiguration

# Tạo môi trường
env = Environment("evi-cem-env")
env.python.conda_dependencies.add_pip_package("torch==1.13.1")
env.python.conda_dependencies.add_pip_package("pytorch-lightning==1.9.0")
env.python.conda_dependencies.add_pip_package("torchvision==0.14.1")
env.python.conda_dependencies.add_pip_package("azureml-core")
env.python.conda_dependencies.add_pip_package("pillow")
env.python.conda_dependencies.add_pip_package("scikit-learn")

# Cấu hình Docker
docker_config = DockerConfiguration(use_docker=True)

# Đẩy score.py và evi_cem.py lên Azure
inference_config = InferenceConfig(entry_script="score.py",
                                   source_directory="/content",  # Bao gồm cả score.py và evi_cem.py
                                   environment=env)

# Cấu hình ACI
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=2)

# Triển khai
try:
    service = Model.deploy(
        workspace=ws,
        name='evi-cem-service4',
        models=[registered_model],
        inference_config=inference_config,
        deployment_config=aci_config,
        overwrite=True
    )
    service.wait_for_deployment(show_output=True)
    print("Deployment successful!")
    print(service.get_logs())
except Exception as e:
    print(f"Deployment failed: {str(e)}")
    if hasattr(service, 'get_logs'):
        print("Logs:", service.get_logs())
scoring_uri = service.scoring_uri  # Thay bằng URL thực tế nếu cần
print("Scoring URI:", scoring_uri)
import requests
import base64
import json

# Lấy URL từ service (chạy sau khi deploy)
scoring_uri = service.scoring_uri  # Thay bằng URL thực tế nếu cần
print("Scoring URI:", scoring_uri)

# Chuẩn bị dữ liệu đầu vào
image_path = "/content/00a61ae0aa6d43a08152a7c4692ef9e2.jpg"  # Đường dẫn tới ảnh của bạn
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
payload = {"image_base64": image_base64}
payload_json = json.dumps(payload)

# Gửi request
headers = {"Content-Type": "application/json"}
response = requests.post(scoring_uri, data=payload_json, headers=headers)

# In kết quả
print("Status code:", response.status_code)
print("Response:", response.text)
