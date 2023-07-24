import os
from pathlib import Path

import gradio as gr # add this line
from tortoise.api import MODELS_DIR
from tortoise.inference import (
  infer_on_texts,
  run_and_save_tts,
  split_and_recombine_text,
)
from tortoise.utils.diffusion import SAMPLERS
from app_utils.conf import TortoiseConfig
from app_utils.funcs import (
  timeit,
  load_model,
  list_voices,
  load_voice_conditionings,
)

LATENT_MODES = [
  "Tortoise original (bad)",
  "average per 4.27s (broken on small files)",
  "average per voice file (broken on small files)",
]

def main():
    
    
    conf = TortoiseConfig()
    
    text = gr.inputs.Textbox(
        label="Text",
        lines=5,
    )
    extra_voices_dir = gr.inputs.Textbox(
        label="Extra Voices Directory",
        default=conf.EXTRA_VOICES_DIR,
        type="text",
    )
    voices, extra_voices_ls = list_voices(extra_voices_dir.value)
    voice = gr.inputs.Dropdown(
        label="Voice",
        choices=voices,
        default=voices[0],
    )
    preset = gr.inputs.Dropdown(
        label="Preset",
        choices=(
            "single_sample",
            "ultra_fast",
            "very_fast",
            "ultra_fast_old",
            "fast",
            "standard",
            "high_quality",
        ),
        default="ultra_fast",
    )
    
    candidates = gr.inputs.Slider(
        label="Candidates",
        minimum=1,
        maximum=3,
        default=1,
        step=1,
    )
    
    latent_averaging_mode = gr.inputs.Radio(
        label="Latent averaging mode",
        choices=LATENT_MODES,
        default=LATENT_MODES[0],

    )
    sampler = gr.inputs.Radio(
        label="Sampler",
        choices=SAMPLERS,
        default=SAMPLERS[1],
    )
    steps = gr.inputs.Slider(
        label="Steps",
        minimum=1,
        maximum=100,
        default=10,
        step=1,
    )
    seed = gr.inputs.Slider(
        label="Seed",
        minimum=-1,
        maximum=1000,
        default=-1,
        step=1,
    )
    voice_fixer = gr.Checkbox(
        label="Voicefixer", 
        value=False,
    )
    cond_free = gr.Checkbox(
        label="Conditioning Free", 
        value=True,
    )
    min_chars_to_split = gr.inputs.Slider(
        label="Min chars to Split",
        minimum=50,
        maximum=1000,
        default=200,
    )
    kv_cache = gr.Checkbox(
        label="Key-Value Cache",
        value=True,
    )
    high_vram = not gr.Checkbox(
        label="Key-Value Cache",
        value=conf.LOW_VRAM,
    )
    
    model_dir = MODELS_DIR
    diff_checkpoint = None
    ar_checkpoint = None
    tts = load_model(model_dir, high_vram, kv_cache, ar_checkpoint, diff_checkpoint)



    def inference(text, extra_voices_dir, voice, preset, candidates, latent_averaging_mode, sampler, steps, seed, kv_cache, cond_free, voice_fixer):
        # add the code for text-to-speech generation from the context page code
        assert latent_averaging_mode
        assert preset
        assert voice
        output_path = "output" 
        def show_generation(fp, filename: str):
            """
            audio_buffer = BytesIO()
            save_gen_with_voicefix(g, audio_buffer, squeeze=False)
            torchaudio.save(audio_buffer, g, 24000, format='wav')
            """
            # return the filepath and the download link as a tuple
            return fp, 


        voices, extra_voices_ls = list_voices(extra_voices_dir)

        selected_voices = voice.split(",")
        
        for k, selected_voice in enumerate(selected_voices):
            output_htmls = []
            if "&" in selected_voice:
                voice_sel = selected_voice.split("&")
            else:
                voice_sel = [selected_voice]
            voice_samples, conditioning_latents = load_voice_conditionings(
                voice_sel, extra_voices_ls
            )

            voice_path = Path(os.path.join(output_path, selected_voice))

            
            with timeit(
                f"Generating {candidates} candidates for voice {selected_voice} (seed={seed})"
            ):
                os.makedirs(output_path, exist_ok=True)

                nullable_kwargs = {
                    k: v
                    for k, v in zip(
                        ["sampler", "diffusion_iterations", "cond_free"],
                        [sampler, steps, cond_free],
                    )
                    if v is not None
                }

                def call_tts(text: str):
                    return tts.tts_with_preset(
                        text,
                        k=int(candidates),
                        voice_samples=voice_samples,
                        conditioning_latents=conditioning_latents,
                        preset=preset,
                        use_deterministic_seed=seed,
                        return_deterministic_state=True,
                        half=False,
                        cvvp_amount=0.0,
                        latent_averaging_mode=LATENT_MODES.index(
                            latent_averaging_mode
                        ),
                        **nullable_kwargs,
                    )

                if len(text) < min_chars_to_split.value:
                    filepaths = run_and_save_tts(
                        call_tts,
                        text,
                        voice_path,
                        return_deterministic_state=True,
                        return_filepaths=True,
                        voicefixer=voice_fixer,
                    )
                else:
                    desired_length = int(min_chars_to_split.value)
                    texts = split_and_recombine_text(
                        text, desired_length, desired_length + 100
                    )
                    filepaths = infer_on_texts(
                        call_tts,
                        texts,
                        voice_path,
                        return_deterministic_state=True,
                        return_filepaths=True,
                        lines_to_regen=set(range(len(texts))),
                        voicefixer=voice_fixer,
                    )
                    
            
                if len(filepaths) < candidates:
                    candidates = len(filepaths)
                
                if len(filepaths) == 1:
                    return filepaths[0], None, None
                elif len(filepaths) == 2:
                    return filepaths[0], filepaths[1], None
                elif len(filepaths) == 3:
                    return filepaths[0], filepaths[1], filepaths[2]
                else:
                    return None, None, #None # or some other default value


    output1 = gr.Audio(label="Output 1", type="filepath")
    output2 = gr.Audio(label="Output 2", type="filepath")
    output3 = gr.Audio(label="Output 3", type="filepath")

    
      # create and launch the interface
    interface = gr.Interface(
        fn=inference, # define the inference function
        inputs=[text, extra_voices_dir, voice, preset, candidates, latent_averaging_mode, sampler, steps, seed, kv_cache, cond_free, voice_fixer], # pass the input components as a list
        outputs=[output1, output2, output3], # pass the output components as a list
        title="Tortoise TTS Fast Demo", # add a title for the interface
        description="A text-to-speech demo using tortoise-tts-fast model.",
      )

    interface.launch(share=True, inline=True, enable_queue=True) # launch the interface and share it with a public link

if __name__ == "__main__":
  main()