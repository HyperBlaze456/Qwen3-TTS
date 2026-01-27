import torch
import gradio as gr
from qwen_tts import Qwen3TTSModel

# 모델 로드
print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    '/mnt/d/리서치/JAXLearn/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
    device_map='cuda:0',
    dtype=torch.bfloat16,
)
print("Model loaded!")

LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]

INSTRUCT_EXAMPLES = [
    "밝고 친근한 20대 여성 목소리",
    "낮고 차분한 중년 남성 목소리",
    "귀엽고 높은 톤의 애니메이션 캐릭터",
    "또렷하고 전문적인 뉴스 앵커 스타일",
    "따뜻하고 부드러운 할머니 목소리",
    "에너지 넘치는 젊은 남성 게이머 목소리",
    "Warm and gentle young female voice",
    "Deep, authoritative male narrator voice",
]


def generate_speech(text: str, language: str, instruct: str):
    if not text.strip():
        return None

    wavs, sr = model.generate_voice_design(
        text=text,
        language=language,
        instruct=instruct,
    )

    return (sr, wavs[0])


with gr.Blocks(title="Qwen3-TTS Voice Design Demo") as demo:
    gr.Markdown("# Qwen3-TTS Voice Design Demo")
    gr.Markdown("텍스트와 원하는 음성 스타일을 입력하면 TTS 음성을 생성합니다.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="텍스트",
                placeholder="음성으로 변환할 텍스트를 입력하세요...",
                lines=3,
            )

            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=LANGUAGES,
                    value="Korean",
                    label="언어",
                )

            instruct_input = gr.Textbox(
                label="음성 스타일 (Instruct)",
                placeholder="원하는 음성 스타일을 설명하세요...",
                lines=2,
            )

            gr.Examples(
                examples=INSTRUCT_EXAMPLES,
                inputs=instruct_input,
                label="스타일 예시",
            )

            generate_btn = gr.Button("음성 생성", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="생성된 음성", type="numpy")

    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, language_dropdown, instruct_input],
        outputs=audio_output,
    )

    gr.Markdown("---")
    gr.Markdown("### 사용 팁")
    gr.Markdown("""
    - **instruct**에 성별, 나이, 톤, 감정 등을 자세히 설명할수록 좋습니다.
    - 지원 언어: 중국어, 영어, 일본어, 한국어, 독일어, 프랑스어, 러시아어, 포르투갈어, 스페인어, 이탈리아어
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
