import os
import torch
from threading import Thread
import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,TextIteratorStreamer,AutoTokenizer
from qwen_vl_utils import process_vision_info
from trellis.pipelines import TrellisImageTo3DPipeline,TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import trimesh
from trimesh.exchange.gltf import export_glb
import numpy as np
import tempfile
import copy
import plotly.graph_objs as go
from PIL import Image
import plotly.express as px
import random
import open3d as o3d

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def token_to_mesh(full_response):
    d1=full_response.split("><mesh")
    d2=[]
    for i in range(len(d1)):
        try:
            if d1[i][:5]=="<mesh":
                d2.append(int(d1[i][5:]))
            else:
                d2.append(int(d1[i]))
        except:
            pass
    while len(d2)<1024:
        d2.append(d2[-1])
    encoding_indices=torch.tensor(d2).unsqueeze(0)
    return encoding_indices

def save_ply_from_array(verts):    
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {verts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]
    tmpf = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmpf.write(("\n".join(header) + "\n").encode("utf-8"))
    np.savetxt(tmpf, verts, fmt="%.6f")
    tmpf.flush(); tmpf.close()
    return tmpf.name

def predict(_chatbot,task_history,viewer_voxel,viewer_mesh,task_new,seed,top_k,top_p,temperature):
    torch.manual_seed(seed)
    chat_query = _chatbot[-1][0]
    query = task_history[-1][0]

    if len(chat_query) == 0:
        _chatbot.pop()
        task_history.pop()
        return _chatbot,task_history,viewer_voxel,viewer_mesh,task_new
    print("User: " + _parse_text(query))
    history_cp = copy.deepcopy(task_history)
    full_response = ""
    messages = []
    content = []

    image_lst = []
    for q, a in task_new:
        if isinstance(q, (tuple, list)):
            if not is_video_file(q[0]):
                image_lst.append(q[0])
            else:
                image_lst.append(q[0])

    task_new.clear()
    for q, a in history_cp:
        if isinstance(q, (tuple, list)):
            if is_video_file(q[0]):
                content.append({'video': f'file://{q[0]}'})
            else:
                content.append({'image': f'file://{q[0]}'})
        else:
            content.append({'text': q})
            messages.append({'role': 'user', 'content': content})
            messages.append({'role': 'assistant', 'content': [{'text': a}]})
            content = []
    messages.pop()
    messages = _transform_messages(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs,videos=video_inputs, padding=True, return_tensors='pt')
    inputs = inputs.to(model.device)

    eos_token_id = [tokenizer.eos_token_id,159858]
    streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {'max_new_tokens': 2048, 'streamer': streamer,"eos_token_id":eos_token_id,\
                   "top_k":top_k,"top_p":top_p,"temperature":temperature,"eos_token_id":eos_token_id,**inputs}

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    full_response = ""
    encoding_indices = None
    _chatbot[-1] = (_parse_text(chat_query), "")  
    for new_text in streamer:
        if new_text:
            if "<mesh" in new_text:
                encoding_indices = token_to_mesh(new_text)
                new_text = new_text.replace("><",",")[1:-1]
                new_text = new_text.split("mesh-start,")[1].split(",mesh-end")[0]
                new_text = f"mesh-start\n{new_text}\nmesh-end"
            full_response += new_text
            _chatbot[-1] = (_parse_text(chat_query), _parse_text(full_response))
            yield _chatbot,viewer_voxel,viewer_mesh,task_new

    task_history[-1] = (chat_query, full_response)
    yield _chatbot,viewer_voxel,viewer_mesh,task_new

    if encoding_indices is not None:
        print("processing mesh...")
        recon = vqvae.Decode(encoding_indices.to(model.device))
        z_s           = recon[0].detach().cpu() 
        z_s           = (z_s>0)*1      
        indices       = torch.nonzero(z_s[0] == 1)  
        position_recon= (indices.float() + 0.5) / 64 - 0.5 
        fig = make_pointcloud_figure(position_recon)
        yield _chatbot,fig,viewer_mesh,task_new

        position=position_recon
        coords        = ((position + 0.5) * 64).int().contiguous()
        ss            = torch.zeros(1, 64, 64, 64, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        ss=ss.unsqueeze(0)
        coords = torch.argwhere(ss>0)[:, [0, 2, 3, 4]].int()
        coords = coords.to(model.device)
        try:
            print("processing mesh...")
            if len(image_lst) == []:
                # text to 3d
                with torch.no_grad():
                    prompt  = chat_query
                    cond    = pipeline_text.get_cond([prompt])
                    slat    = pipeline_text.sample_slat(cond, coords)
                    outputs = pipeline_text.decode_slat(slat, ['mesh', 'gaussian'])

                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,          
                    texture_size=1024,    
                    verbose=False  
                )
                glb.export(f"temper.glb")
                print("processing mesh over...")
                yield _chatbot,fig,"temper.glb"
            else:
                # image to 3d
                with torch.no_grad():
                    img = pipeline_image.preprocess_image(Image.open(image_lst[-1]))
                    cond    = pipeline_image.get_cond([img])
                    slat    = pipeline_image.sample_slat(cond, coords)
                    outputs = pipeline_image.decode_slat(slat, ['mesh', 'gaussian'])
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.95,          
                    texture_size=1024,    
                    verbose=False  
                )
                glb.export(f"temper.glb")
                print("processing mesh over...")
                yield _chatbot,fig,"temper.glb",task_new
        except:
            print("processing mesh...bug")
            yield _chatbot,fig,viewer_mesh,task_new

def regenerate(_chatbot, task_history):
    if not task_history:
        return _chatbot
    item = task_history[-1]
    if item[1] is None:
        return _chatbot
    task_history[-1] = (item[0], None)
    chatbot_item = _chatbot.pop(-1)
    if chatbot_item[0] is None:
        _chatbot[-1] = (_chatbot[-1][0], None)
    else:
        _chatbot.append((chatbot_item[0], None))
    _chatbot_gen = predict(_chatbot, task_history)
    for _chatbot in _chatbot_gen:
        yield _chatbot

def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def add_text_prefix(text):
    text = f"Please generate a 3D asset based on the prompt I provided: {text}"
    return gr.update(value=text)

def token_to_words(token):
    mesh             = "<mesh-start>"
    for j in range(1024):
        mesh += f"<mesh{token[j]}>"
    mesh            += "<mesh-end>"
    return mesh

def add_text(history, task_history, text,task_new):
    task_text = text
    history = history if history is not None else []
    task_history = task_history if task_history is not None else []
    history = history + [(_parse_text(text), None)]
    task_history = task_history + [(task_text, None)]
    task_new     = task_new + [(task_text, None)]
    return history, task_history,task_new

def add_file(history, task_history, file, task_new, fig, query):
    if file.name.endswith(('.obj', '.glb')):
        position_recon = load_vertices(file.name)#(N,3)

        coords        = ((torch.from_numpy(position_recon) + 0.5) * 64).int().contiguous()
        ss            = torch.zeros(1, 64, 64, 64, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        token         = vqvae.Encode(ss.to(dtype=torch.float32).unsqueeze(0).to("cuda"))
        token         = token[0].cpu().numpy().tolist()
        words         = token_to_words(token)
        fig            = make_pointcloud_figure(position_recon,rotate=True)
        return history, task_history,file.name,task_new,fig,gr.update(
            value=f"{words}\nGive a quick overview of the object represented by this 3D mesh.")
    history = history if history is not None else []
    task_history = task_history if task_history is not None else []
    history = history + [((file.name,), None)]
    task_history = task_history + [((file.name,), None)]
    task_new     = task_new + [((file.name,), None)]
    return history, task_history, file.name, task_new, fig, query

def reset_user_input():
    return gr.update(value="")

def reset_state(task_history):
    task_history.clear()
    return []

def make_pointcloud_figure(verts,rotate=False):
    if rotate:
        verts = verts.copy()
        verts[:, 0] *= -1.0
    N      = len(verts)
    soft_palette = ["#FFEBEE", "#FFF3E0", "#FFFDE7", "#E8F5E9",]
    palette = px.colors.qualitative.Set3
    base_colors = [palette[i % len(palette)] for i in range(N)]
    random.shuffle(base_colors)

    camera = dict(
        eye=dict(x=0.0, y=2.5, z=0.0),   
        center=dict(x=0.0, y=0.0, z=0.0),
        up=dict(x=0.0, y=0.0, z=1.0),    
        projection=dict(type="orthographic")  
    )

    scatter = go.Scatter3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        mode='markers',
        marker=dict(
            size=2,           
            color=base_colors,  
            opacity=1,        
            line=dict(width=1)  
        )
    )
    layout = go.Layout(
        width =700,    
        height=200,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig = go.Figure(data=[scatter], layout=layout)
    return fig

def rotate_points(points, axis='x', angle_deg=90):
    angle_rad = np.deg2rad(angle_deg)
    if axis == 'x':
        R = trimesh.transformations.rotation_matrix(angle_rad, [1, 0, 0])[:3, :3]
    elif axis == 'y':
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])[:3, :3]
    elif axis == 'z':
        R = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])[:3, :3]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return points @ R.T

def convert_trimesh_to_open3d(trimesh_mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(trimesh_mesh.vertices, dtype=np.float64)
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(trimesh_mesh.faces, dtype=np.int32)
    )
    return o3d_mesh

def load_vertices(filepath):
    mesh = trimesh.load(filepath, force='mesh')
    mesh = convert_trimesh_to_open3d(mesh)
    vertices = np.asarray(mesh.vertices)
    min_vals = vertices.min()
    max_vals = vertices.max()
    vertices_normalized = (vertices - min_vals) / (max_vals - min_vals)  
    vertices = vertices_normalized * 1.0 - 0.5  
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    voxel = rotate_points(vertices, axis='x', angle_deg=90)
    return voxel

def add_file2(history, task_history, file,task_new):
    history = history if history is not None else []
    task_history = task_history if task_history is not None else []
    history = history + [((file,), None)]
    task_history = task_history + [((file,), None)]
    task_new     = task_new + [((file,), None)]
    return history, task_history,file,task_new

def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video' in item:
                new_item = {'type': 'video', 'video': item['video']}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages

from trellis.models.sparse_structure_vqvae import VQVAE3D
device       = torch.device("cuda")
vqvae        = VQVAE3D(num_embeddings=8192)
device       = torch.device("cuda")
vqvae.eval()
filepath = hf_hub_download(repo_id="yejunliang23/3DVQVAE",filename="3DVQVAE.bin")
state_dict = torch.load(filepath, map_location="cpu")
vqvae.load_state_dict(state_dict)
vqvae=vqvae.to(device)

MODEL_DIR = "yejunliang23/ShapeLLM-7B-omni"
model_ckpt_path=MODEL_DIR
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_ckpt_path, torch_dtype="auto", device_map={"": 0})
processor = AutoProcessor.from_pretrained(model_ckpt_path)
tokenizer = processor.tokenizer
from huggingface_hub import hf_hub_download

pipeline_text = TrellisTextTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")
pipeline_text.to(device)
pipeline_image = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline_image.to(device)

_DESCRIPTION = '''
* Project page of ShapeLLM-Omni: https://ml.cs.tsinghua.edu.cn/~zhengyi/CRM/
* As generation tasks currently lack support for multi-turn dialogue, it’s strongly recommended to clear the chat history before starting a new task
* The model's 3D understanding is limited to shape only, so color and texture should be ignored in 3D captioning tasks
'''
with gr.Blocks() as demo:
    gr.Markdown("# ShapeLLM-omni: A Native Multimodal LLM for 3D Generation and Understanding")
    gr.Markdown(_DESCRIPTION)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label='ShapeLLM-Omni', elem_classes="control-height", height=500)
            seed        = gr.Number(value=42, label="seed", precision=0)
            top_k       = gr.Slider(label="top_k",minimum=1024,maximum=8194,value=1024,step=10)
            top_p       = gr.Slider(label="top_p",minimum=0.1,maximum=1.0,value=0.1,step=0.05)
            temperature = gr.Slider(label="temperature",minimum=0.1,maximum=1.0,value=0.1,step=0.05)

            query = gr.Textbox(lines=2, label='Input')
            image_input = gr.Image(visible=False, type="filepath", label="Image Input")
            with gr.Column():
                with gr.Row():
                    addfile_btn = gr.UploadButton("📁 Upload", file_types=["image", "video",".obj",".glb"])
                    submit_btn = gr.Button("🚀 Submit")
                with gr.Row():
                    regen_btn = gr.Button("🤔️ Regenerate")
                    empty_bin = gr.Button("🧹 Clear History")
            task_history = gr.State([])
            task_new     = gr.State([])
        with gr.Column():
            viewer_plot  = gr.Plot(label="Voxel Visual",scale=1.0)
            viewer_mesh  = gr.Model3D(label="Mesh Visual", height=200,scale=1.0)

            examples_text = gr.Examples(
                examples=[
                    ["A drone with four propellers and a central body."],
                    ["A stone axe with a handle."],
                    ["the titanic, aerial view."],
                    ["A 3D model of a small yellow and blue robot with wheels and two pots."],
                    ["A futuristic vehicle with a sleek design and multiple wheels."],
                    ["A car with four wheels and a roof."],
                ],
                inputs=[query],
                label="text-to-3d examples",
                fn=add_text_prefix,
                outputs=[query],
                cache_examples=True,
                )

            examples_text.dataset.click(
                fn=add_text,
                inputs=[chatbot, task_history, query,task_new],
                outputs=[chatbot, task_history,task_new],
            )
            examples_image = gr.Examples(
                label="image-to-3d examples",
                examples=[os.path.join("examples", i) for i in os.listdir("examples")],
                inputs=[image_input],
                examples_per_page = 20,
            )
            image_input.change(
                fn=add_file2,
                inputs=[chatbot, task_history, image_input,task_new],
                outputs=[chatbot, task_history,viewer_mesh,task_new],
                show_progress=True
            )

    submit_btn.click(add_text, [chatbot, task_history, query,task_new],\
                               [chatbot, task_history,task_new]).then(
        predict, [chatbot, task_history,viewer_plot,viewer_mesh,task_new,seed,top_k,top_p,temperature],\
                 [chatbot,viewer_plot,viewer_mesh,task_new], show_progress=True
    )
    submit_btn.click(reset_user_input, [], [query])
    empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
    regen_btn.click(regenerate,  [chatbot, task_history], [chatbot], show_progress=True)
    addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn, task_new, viewer_plot, query],\
                                 [chatbot, task_history, viewer_mesh, task_new, viewer_plot, query],\
                                  show_progress=True)
    

demo.launch()
