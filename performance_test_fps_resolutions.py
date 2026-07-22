import torch
import torch.nn as nn
import time
import os
import gc
import subprocess
import logging
import threading
import torchvision.models as models
import timm  # for RepVGG
from fvcore.nn import FlopCountAnalysis, flop_count_table  # for real FLOPs measurement

torch.backends.cudnn.benchmark = True

# ==========================================
# HELPER FUNCTION: BACKGROUND HARDWARE MONITOR
# ==========================================
class HardwareMonitor:
    def __init__(self):
        self.keep_running = False
        self.powers = []
        self.utils = []
        self.temps = []
        self.thread = None

    def _monitor(self):
        while self.keep_running:
            try:
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw', 
                     '--format=csv,noheader,nounits'], encoding='utf-8')
                u, t, p = result.strip().split(',')
                self.utils.append(float(u))
                self.temps.append(float(t))
                self.powers.append(float(p))
            except:
                pass
            time.sleep(0.1) 

    def start(self):
        self.keep_running = True
        self.powers, self.utils, self.temps = [] , [], []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.keep_running = False
        if self.thread:
            self.thread.join()
        
        if self.powers:
            avg_p = sum(self.powers) / len(self.powers)
            avg_u = sum(self.utils) / len(self.utils)
            max_t = max(self.temps) 
            return f"{int(avg_u)}%", f"{int(max_t)}C", f"{avg_p:.1f}W"
        return "N/A", "N/A", "N/A"

# ==========================================
# 1. SETTINGS (CONFIG)
# ==========================================
MODEL_PATH = "results/BEST_MODEL.pth" 
BATCH_SIZES = [1, 2,4,8, 16, 32, 64, 128, 256] 
INPUT_SHAPE = (3, 224, 224) 
NUM_CLASSES = 100
WARMUP_ROUNDS = 50
TEST_ROUNDS = 200

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. MODEL LOADERS
# ==========================================
def get_custom_halsp_model(path=None):
    try:
        from halsp import ResNet50
        model = ResNet50(num_classes=NUM_CLASSES).to(device)
        if path and os.path.exists(path):
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
        return model, "HalspNet"
    except ImportError:
        logger.error("ERROR: 'halsp.py' not found!")
        return None, None

def get_standard_torchvision_model():
    model = models.resnet50(weights=None, num_classes=NUM_CLASSES)
    return model.to(device), "ResNet-50 (Raw)"

def get_efficientnet():
    effnet = models.efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
    return effnet.to(device), "EfficientNet-B0 (Raw)"

def get_mobilenet():
    mobilenet = models.mobilenet_v3_large(weights=None, num_classes=NUM_CLASSES)
    return mobilenet.to(device), "MobileNetV3-Large (Raw)"

def get_shufflenet():
    shufflenet = models.shufflenet_v2_x1_0(weights=None, num_classes=NUM_CLASSES)
    return shufflenet.to(device), "ShuffleNetV2-1.0x (Raw)"

def get_convnext():
    convnext = models.convnext_tiny(weights=None, num_classes=NUM_CLASSES)
    return convnext.to(device), "ConvNeXt-Tiny (Raw)"

def get_repvgg():
    # RepVGG must be switched to deploy mode for inference speed measurement
    repvgg = timm.create_model('repvgg_a0', pretrained=False, num_classes=NUM_CLASSES)
    if hasattr(repvgg, 'switch_to_deploy'):
        repvgg.switch_to_deploy()
    return repvgg.to(device), "RepVGG-A0 (Deploy Mode)"

def get_ghostnet():
    ghostnet = timm.create_model('ghostnet_100', pretrained=False, num_classes=NUM_CLASSES)
    return ghostnet.to(device), "GhostNet-100 (Raw)"

def get_mobilevit():
    mobilevit = timm.create_model('mobilevit_xs', pretrained=False, num_classes=NUM_CLASSES)
    return mobilevit.to(device), "MobileViT-XS (Raw)"

def get_regnet():
    regnet = models.regnet_y_400mf(weights=None, num_classes=NUM_CLASSES)
    return regnet.to(device), "RegNetY-400MF (Raw)"

# ==========================================
# 3. BENCHMARK ENGINE
# ==========================================
def run_inference_benchmark(model, model_name):
    # Put model in eval mode directly
    model.eval() 

    logger.info(f"\n{'='*80}\nMODEL: {model_name} BEING ANALYZED\n{'='*80}")
    
    dummy_single = torch.randn(1, *INPUT_SHAPE).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3): _ = model(dummy_single)
            
    # Precise FLOPs and Parameter measurement with fvcore
    try:
        flops_obj = FlopCountAnalysis(model, dummy_single)
        flops_obj.unsupported_ops_warnings(False) # Hide unnecessary warnings
        flops = flops_obj.total()
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"[+] Params: {params/1e6:.2f}M | FLOPs: {flops/1e9:.4f}G")
    except Exception as e:
        logger.error(f"[!] fvcore FLOPs Error: {e}")
        flops, params = 0, 0

    logger.info(f"\n{'Batch':<6} | {'Throughput':<15} | {'Latency':<10} | {'VRAM':<10} | {'GPU %':<7} | {'Temp':<6} | {'Power':<8}")
    logger.info("-" * 85)
    
    batch_results = []
    
    for b_size in BATCH_SIZES:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
        dummy_input = torch.randn(b_size, *INPUT_SHAPE).to(device)
        
        try:
            with torch.inference_mode():
                for _ in range(WARMUP_ROUNDS): _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                
            monitor = HardwareMonitor()
            monitor.start()
            
            if torch.cuda.is_available():
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                
                start_evt.record()
                with torch.inference_mode():
                    for _ in range(TEST_ROUNDS): _ = model(dummy_input)
                end_evt.record()
                torch.cuda.synchronize()
                total_time_ms = start_evt.elapsed_time(end_evt)
            else:
                start_time = time.time()
                with torch.inference_mode():
                    for _ in range(TEST_ROUNDS): _ = model(dummy_input)
                total_time_ms = (time.time() - start_time) * 1000

            gpu_util, gpu_temp, gpu_power = monitor.stop()
            
            avg_latency = total_time_ms / TEST_ROUNDS
            throughput = (TEST_ROUNDS * b_size) / (total_time_ms / 1000)
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0
            
            logger.info(f"{b_size:<6} | {throughput:<15.2f} | {avg_latency:<10.2f} | {vram_mb:<10.2f} | {gpu_util:<7} | {gpu_temp:<6} | {gpu_power:<8}")
            batch_results.append((b_size, throughput, avg_latency, vram_mb, gpu_util, gpu_temp, gpu_power))
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"{b_size:<6} | {'OOM':<15} | {'-':<10} | {'-':<10} | {'-':<7} | {'-':<6} | {'-':<8}")
                batch_results.append((b_size, "OOM", "-", "-", "-", "-", "-"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            else: raise e

    return {"name": model_name, "params": params, "flops": flops, "results": batch_results}

# ==========================================
# 4. MAIN FLOW
# ==========================================
if __name__ == "__main__":
    final_reports = []
    
    # LAZY LOADING - To prevent memory overflow
    model_loaders = [
        lambda: get_custom_halsp_model(MODEL_PATH),
        get_standard_torchvision_model,
        get_efficientnet,
        get_mobilenet,
        get_shufflenet,
        get_convnext,
        get_repvgg,
        get_ghostnet,
        get_mobilevit,
        get_regnet,
    ]

    for loader in model_loaders:
        m, name = loader()
        
        if m is not None:
            rep = run_inference_benchmark(m, name)
            final_reports.append(rep)
            
            del m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

    # --- REPORTING ---
    report_file = "inference_comparison_report_hw.txt"
    with open(report_file, "w") as f:
        f.write("--- COMPREHENSIVE INFERENCE PERFORMANCE REPORT (224x224 INPUT) ---\n\n")
        for rep in final_reports:
            f.write(f"MODEL: {rep['name']}\n")
            f.write(f"Params: {rep['params']/1e6:.2f} M | Forward FLOPs: {rep['flops']/1e9:.4f} G\n")
            f.write(f"{'Batch':<6} | {'TP (img/s)':<15} | {'Lat (ms)':<10} | {'VRAM (MB)':<10} | {'GPU%':<7} | {'Temp':<6} | {'Power':<8}\n")
            f.write("-" * 85 + "\n")
            for r in rep["results"]:
                b, t, l, v, gu, gt, gp = r
                if t == "OOM":
                    f.write(f"{b:<6} | {'OOM':<15} | {'-':<10} | {'-':<10} | {'-':<7} | {'-':<6} | {'-':<8}\n")
                else:
                    f.write(f"{b:<6} | {t:<15.2f} | {l:<10.2f} | {v:<10.2f} | {gu:<7} | {gt:<6} | {gp:<8}\n")
            f.write("\n" + "="*85 + "\n\n")

    logger.info(f"\n>>> Comparative inference report saved successfully as '{report_file}'.")
