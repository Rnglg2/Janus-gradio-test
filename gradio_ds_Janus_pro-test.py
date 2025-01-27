import os
import gradio as gr
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

def initialize_models():
    model_path = "deepseek-ai/Janus-Pro-7B" 
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True
    ).to(torch.bfloat16).cuda().eval()
    
    return vl_gpt, vl_chat_processor, tokenizer

vl_gpt, processor, tokenizer = initialize_models()

@torch.inference_mode()
def generate_image(
    prompt: str,
    temperature: float = 1.0,
    parallel_size: int = 4,
    cfg_weight: float = 5.0,
    img_size: int = 384,
    patch_size: int = 16,
    image_token_num_per_image: int = 576
):
    conversation = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    full_prompt = sft_format + processor.image_start_tag

    input_ids = processor.tokenizer.encode(full_prompt)
    input_ids = torch.LongTensor(input_ids).cuda()
    
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    
    outputs = None
    for i in range(image_token_num_per_image):
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        
        next_token = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(1)

    dec = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(torch.int),
        shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    
    return [PIL.Image.fromarray(img) for img in dec]

@torch.inference_mode()
def generate_response(image, question):
    if image is None or not question:
        return "请先上传图片"
    
    conversation = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{question}", "images": [image]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [PIL.Image.fromarray(image)]
    prepare_inputs = processor(
        conversations=conversation, 
        images=pil_images, 
        force_batchify=True
    ).to(vl_gpt.device)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    
    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

with gr.Blocks(theme=gr.themes.Soft(), title="Janus-Pro") as demo:
    gr.Markdown("# 🎨 Janus-Pro")
    with gr.Tabs():
        with gr.Tab("图像生成"):
            with gr.Row():
                with gr.Column():
                    image_prompt = gr.Textbox(
                        label="图像描述",
                        placeholder="例：A beautiful princess from Kabul, dressed in traditional red and white dress",
                        lines=3
                    )
                    with gr.Accordion("高级参数", open=False):
                        image_temp = gr.Slider(0.1, 2.0, value=1.0, label="温度系数")
                        cfg_weight = gr.Slider(1.0, 10.0, value=5.0, label="引导权重")
                        num_images = gr.Slider(1, 8, value=4, step=1, label="生成数量")
                    image_btn = gr.Button("开始生成", variant="primary")
                
                with gr.Column():
                    image_gallery = gr.Gallery(
                        label="生成结果",
                        columns=4,
                        object_fit="contain",
                        height="auto"
                    )
        with gr.Tab("图片问答"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="上传图片", type="numpy")
                    question_input = gr.Textbox(
                        label="输入问题",
                        placeholder="关于这张图片，您有什么想问的？",
                        lines=3
                    )
                    qa_btn = gr.Button("提交问题", variant="primary")
                
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="模型回答",
                        interactive=False,
                        lines=5
                    )
    image_btn.click(
        generate_image,
        inputs=[image_prompt, image_temp, num_images, cfg_weight],
        outputs=image_gallery
    )
    
    qa_btn.click(
        generate_response,
        inputs=[input_image, question_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )