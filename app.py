import gradio as gr
import torch
from PIL import Image
import os
from datetime import datetime
import uuid
import numpy as np
import cv2 # 用于处理视频帧，确保已安装 opencv-python

# 假设 BEN2 模块已经通过 `pip install .` 或 `pip install git+...` 安装
# 这样就可以从 `ben2` 包中导入 `BEN_Base` 和 `AutoModel`
from ben2 import BEN_Base, AutoModel 


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
# 优先使用 BEN_Base 从本地检查点加载，因为这是您仓库中的推荐方式
try:
    model = BEN_Base().to(device).eval()
    # 假设 BEN2_Base.pth 文件在 app.py 所在的目录
    # 您可以根据实际情况修改此路径
    model.loadcheckpoints("./BEN2_Base.pth") 
    print("模型已从本地检查点 BEN2_Base.pth 加载。")
except Exception as e:
    print(f"尝试从本地检查点加载模型失败: {e}")
    print("尝试从 Hugging Face 加载模型...")
    try:
        model = AutoModel.from_pretrained("PramaLLC/BEN2").to(device).eval()
        print("模型已从 Hugging Face 加载。")
    except Exception as e_hf:
        print(f"从 Hugging Face 加载模型也失败: {e_hf}")
        print("请检查网络连接或模型路径。")
        raise RuntimeError("无法加载 BEN2 模型。请检查您的安装和模型文件。")


# 创建输出目录
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def generate_unique_filename(base_filename):
    """
    生成一个包含时间戳和 UUID 的唯一文件名，以防止文件重复。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8] # 取 UUID 的前8位
    name, ext = os.path.splitext(os.path.basename(base_filename))
    return f"{timestamp}_{unique_id}_{name}{ext}"

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.float16) # 如果有GPU，开启自动混合精度
def process_image(input_image_np, refine_foreground):
    """
    处理单张图片并返回前景图。
    input_image_np 是 Gradio 提供的 numpy 数组格式的图像。
    """
    if input_image_np is None:
        return None, "请上传一张图片！"

    # 将 numpy 数组转换为 PIL.Image
    pil_image = Image.fromarray(input_image_np.astype(np.uint8))

    # 调用 BEN2 模型的推理方法
    # model.inference 方法签名：inference(self, image, refine_foreground=False)
    try:
        foreground_image = model.inference(pil_image, refine_foreground=refine_foreground)
    except Exception as e:
        return None, f"图片处理失败: {e}"

    # 保存输出图像
    original_filename = "image_input.png" # 默认名，可以根据需要从原始上传名中提取
    output_filename = generate_unique_filename(original_filename)
    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    foreground_image.save(output_path)

    return output_path, "图片处理完成！"

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.float16) # 如果有GPU，开启自动混合精度
def process_video(input_video_path, refine_foreground, output_format, rgb_background_color_hex):
    """
    处理视频并返回前景视频。
    input_video_path 是 Gradio 提供的视频文件路径。
    rgb_background_color_hex 是 Gradio ColorPicker 返回的 HEX 字符串。
    """
    if input_video_path is None:
        return None, "请上传一个视频文件！"

    # 解析 RGB 颜色
    # Gradio ColorPicker 返回的是 #RRGGBB 格式的字符串
    rgb_value = tuple(int(rgb_background_color_hex[i:i+2], 16) for i in (1, 3, 5))

    is_webm = (output_format == "WebM (带透明度)")

    # BEN2 的 segment_video 方法会将文件保存到 output_path 指定的目录
    # 我们为每个视频处理创建一个唯一的子目录，以避免文件名冲突
    video_output_sub_dir = os.path.join(OUTPUTS_DIR, generate_unique_filename("video_output"))
    os.makedirs(video_output_sub_dir, exist_ok=True)

    try:
        model.segment_video(
            video_path=input_video_path,
            output_path=video_output_sub_dir, # 这是输出目录
            refine_foreground=refine_foreground,
            webm=is_webm,
            rgb_value=rgb_value,
            fps=0, # 保持原始视频帧率
            batch=1, # 默认批量大小，可根据GPU显存调整
            print_frames_processed=True # 在控制台打印处理进度
        )
    except Exception as e:
        # 清理可能生成的临时目录
        if os.path.exists(video_output_sub_dir):
            try:
                os.rmdir(video_output_sub_dir) # 尝试删除空目录
            except OSError:
                pass # 如果非空，则不删除
        return None, f"视频处理失败: {e}"

    # 构造输出视频的完整路径
    # BEN2.py 中的 segment_video 函数会生成 foreground.webm 或 foreground.mp4
    # 如果有音频，会生成 foreground_output_with_audio.mp4
    
    # 检查是否生成了带音频的 MP4
    output_video_filename_with_audio = os.path.join(video_output_sub_dir, "foreground_output_with_audio.mp4")
    if not is_webm and os.path.exists(output_video_filename_with_audio):
        output_video_path = output_video_filename_with_audio
    else:
        output_video_path = os.path.join(video_output_sub_dir, "foreground.webm" if is_webm else "foreground.mp4")
    
    if not os.path.exists(output_video_path):
        return None, f"视频处理成功，但未找到输出文件: {output_video_path}"

    return output_video_path, "视频处理完成！"


# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# BEN2: 背景擦除网络")
    gr.Markdown("通过 BEN2 模型进行图像或视频的前景分割和背景擦除。")

    with gr.Tab("图像处理"):
        with gr.Row():
            with gr.Column():
                # 移除 tool="editor" 参数以兼容旧版本 Gradio
                image_input = gr.Image(type="numpy", label="上传您的图片") 
                image_refine_checkbox = gr.Checkbox(label="优化前景", value=False, info="此步骤会增加推理时间，但有助于改善抠图边缘。")
                image_button = gr.Button("开始处理图片")
            with gr.Column():
                image_output = gr.Image(label="处理后的前景图片", type="filepath", interactive=False)
                image_status_text = gr.Textbox(label="状态", interactive=False)
        image_button.click(
            process_image,
            inputs=[image_input, image_refine_checkbox],
            outputs=[image_output, image_status_text]
        )
    
    with gr.Tab("视频处理"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="上传您的视频文件")
                video_refine_checkbox = gr.Checkbox(label="优化前景", value=False, info="此步骤会增加推理时间，但有助于改善抠图边缘。")
                video_output_format = gr.Radio(
                    ["WebM (带透明度)", "MP4 (带背景色)"],
                    label="选择输出视频格式",
                    value="MP4 (带背景色)",
                    info="WebM 格式支持透明度（部分播放器可能不支持），而 MP4 将把前景合成到指定背景色上。"
                )
                video_rgb_color_picker = gr.ColorPicker(
                    label="MP4 背景色 (仅MP4格式有效)",
                    value="#00FF00", # 默认绿色背景
                    info="如果输出格式选择 MP4，此颜色将作为视频背景色。"
                )
                video_button = gr.Button("开始处理视频")
            with gr.Column():
                video_output = gr.Video(label="处理后的前景视频", interactive=False)
                video_status_text = gr.Textbox(label="状态", interactive=False)

        video_button.click(
            process_video,
            inputs=[video_input, video_refine_checkbox, video_output_format, video_rgb_color_picker],
            outputs=[video_output, video_status_text]
        )

    gr.Markdown("---")
    gr.Markdown("更多信息请访问 [BEN2 官方网站](https://backgrounderase.net) 或 [Hugging Face 仓库](https://huggingface.co/PramaLLC/BEN2)。")

# 启动 Gradio 应用程序
# server_port=7860 将设置默认访问端口为 7860
# show_api=False 隐藏 Gradio API 文档链接
# share=False 不创建公共分享链接 (如果您在本地运行)
demo.launch(server_port=7860, show_api=False, share=False)