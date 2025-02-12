import importlib.util
import sys
import subprocess

# Required packages and versions
required_packages = {
    "tensorflow": "2.10.0",
    "numpy": "1.23.5",
    "scipy": "1.9.3",
    "matplotlib": "3.6.0"
}

# Function to check if a package is installed
def check_and_install(package, version):
    try:
        pkg = importlib.util.find_spec(package)
        if pkg is None:
            raise ImportError
        else:
            print(f"✅ {package} is already installed.")
    except ImportError:
        print(f"⚠️ {package} not found! Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])

# Check each required package
for pkg, ver in required_packages.items():
    check_and_install(pkg, ver)

# Check CUDA and cuDNN versions
try:
    import tensorflow as tf
    print("\n✅ TensorFlow Installed Version:", tf.__version__)
    print("✅ GPU Available:", tf.config.list_physical_devices('GPU'))

    # Print CUDA and cuDNN versions detected by TensorFlow
    cuda_version = tf.sysconfig.get_build_info().get("cuda_version", "Not Found")
    cudnn_version = tf.sysconfig.get_build_info().get("cudnn_version", "Not Found")
    print(f"✅ CUDA Version: {cuda_version}")
    print(f"✅ cuDNN Version: {cudnn_version}")

except Exception as e:
    print(f"❌ TensorFlow GPU is not working properly: {e}")
